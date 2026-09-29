"""Optional TVM backend. Everything that needs TVM is skipped when `tvm` cannot be imported."""
import numpy as np
import pytest

from qfuzz import tvm_runner
from qfuzz.campaign import summarize
from qfuzz.case import Case, NodeSpec, TensorSpec
from qfuzz.runner import OrtResult

needs_tvm = pytest.mark.skipif(not tvm_runner.available(), reason="apache-tvm not installed")


def _qdq(x, scale, zp_dtype="int8", zp=0, attrs=None, zp_shape=()):
    attrs = attrs or {}
    return Case([NodeSpec("QuantizeLinear", ["x", "s", "z"], ["q"], dict(attrs)),
                 NodeSpec("DequantizeLinear", ["q", "s", "z"], ["y"], dict(attrs))],
                [TensorSpec("x", "float32", np.asarray(x, np.float32))],
                [TensorSpec("s", "float32", np.asarray(scale, np.float32)),
                 TensorSpec("z", zp_dtype, np.full(zp_shape, zp))], ["y"], {"pattern": "test"})


def test_import_without_tvm_is_harmless():
    assert isinstance(tvm_runner.available(), bool)
    assert set(tvm_runner.PATTERNS) and "qlinear_matmul" not in tvm_runner.PATTERNS
    r = OrtResult("tvm", error="ValueError: DequantizeLinear blocked quantization is not supported yet.", stage="import")
    assert tvm_runner.classify_error(r) == "tvm-unsupported"
    assert tvm_runner.classify_error(OrtResult("tvm", error="InternalError: size mismatch", stage="import")) \
        == "tvm-error-import"


def test_summarize_tvm_backend_fields():
    recs = [{"seed": 1, "pattern": "quantize", "tags": [], "ms": 1,
             "verdict": {"cls": "exact", "levels": [], "signature": "exact", "detail": {},
                         "stats": {"tvm_eq_ort_disable": True}}},
            {"seed": 2, "pattern": "quantize", "tags": [], "ms": 1,
             "verdict": {"cls": "tvm-unsupported", "levels": ["tvm"], "signature": "tvm-unsupported|x",
                         "detail": {}, "stats": {}}}]
    s = summarize(recs, 1.0, 1, 0, backend="tvm")
    assert s["config"]["levels"] == ["tvm"]
    assert s["tvm_vs_ort_disable"] == {"bit_identical": 1, "differ": 0, "not_compared": 1}
    assert s["signatures"][0]["cls"] == "tvm-unsupported"


@needs_tvm
def test_tvm_qdq_matches_spec_including_ties_and_saturation():
    # 1.25/0.5 = 2.5 -> 2 (half to even); +inf and 3e9 must saturate to 127, -inf to -128
    c = _qdq([1.25, 1.75, np.inf, 3e9, -np.inf], 0.5, zp=0)
    r = tvm_runner.run(c)
    assert r.error is None, r.error
    np.testing.assert_array_equal(r.outputs["y"], np.array([1.0, 2.0, 63.5, 63.5, -64.0], np.float32))
    assert tvm_runner.judge(c).cls == "exact"


@needs_tvm
def test_tvm_scalar_input_keeps_rank():
    c = _qdq(np.float32(2.4), 1.0)
    r = tvm_runner.run(c)
    assert r.error is None, r.error
    assert r.outputs["y"].shape == () and float(r.outputs["y"]) == 2.0


@needs_tvm
def test_tvm_blocked_quantization_is_explicitly_unsupported():
    c = _qdq(np.ones((2, 2)), np.ones((1, 2)), attrs={"axis": 0, "block_size": 2}, zp_shape=(1, 2))
    v = tvm_runner.judge(c, with_ort=False)
    assert v.cls == "tvm-unsupported", v.to_json()


@needs_tvm
def test_tvm_int4_initializer_import_error():
    c = Case([NodeSpec("DequantizeLinear", ["w", "s"], ["y"])], [],
             [TensorSpec("w", "int4", np.array([1, -2, 3, -4])), TensorSpec("s", "float32", np.float32(1.0))],
             ["y"], {"pattern": "test"})
    v = tvm_runner.judge(c, with_ort=False)
    # TVM 0.27 cannot import int4 initializers (packed vs unpacked byte count). If this starts
    # passing, re-run the TVM campaign: dq_matmul_nbits becomes testable.
    if v.cls == "exact":
        pytest.skip("int4 initializers import fine on this TVM version")
    assert v.cls == "tvm-error-import" and "size mismatch" in v.detail["error"], v.to_json()


def test_tvm_judge_flags_injected_error(monkeypatch):
    """The TVM verdict goes through the same exact-reference oracle: a 1-LSB error far from a tie is caught."""
    c = _qdq([1.0, 2.0, 3.0], 1.0)  # int8 Q->DQ, exact result [1, 2, 3]

    def fake_run(case):
        return OrtResult("tvm", outputs={"y": np.array([1.0, 2.0, 4.0], np.float32)})

    monkeypatch.setattr(tvm_runner, "run", fake_run)
    v = tvm_runner.judge(c, with_ort=False)
    assert v.cls in ("mismatch", "off-by-one") and v.levels == ["tvm"], v.to_json()
    monkeypatch.setattr(tvm_runner, "run", lambda case: OrtResult("tvm", outputs={"y": np.array([1, 2, 3], np.float32)}))
    assert tvm_runner.judge(c, with_ort=False).cls == "exact"


def test_committed_tvm_results_render_and_are_fully_triaged():
    import json
    from pathlib import Path
    from qfuzz.report import render
    root = Path(__file__).resolve().parents[1]
    camp = json.loads((root / "results" / "tvm_campaign.json").read_text())
    assert camp["env"]["backend"] == "tvm" and camp["config"]["cases"] == sum(camp["counts"].values())
    assert "Apache TVM" in render(camp, {})
    tri = json.loads((root / "results" / "tvm_triage.json").read_text())
    assert tri["untriaged"] == []
    assert sum(g["hits"] for g in tri["groups"].values()) + len(tri["findings"]) == sum(
        s["count"] for s in camp["signatures"])
