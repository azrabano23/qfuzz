"""Oracle plumbing: comparison/classification and ORT execution."""
import numpy as np

from qfuzz import reference as ref
from qfuzz.case import Case, NodeSpec, TensorSpec
from qfuzz.oracle import compare, judge, scale_regime
from qfuzz.runner import run_all


def V(value, slack, dtype="uint8"):
    return ref.Val(np.asarray(value, dtype=np.float64 if dtype == "float32" else np.int64),
                   np.asarray(slack, np.float64), dtype)


def test_compare_classes():
    assert compare(np.array([1, 2], np.uint8), V([1, 2], [0, 0])).cls == "exact"
    assert compare(np.array([1, 3], np.uint8), V([1, 2], [0, 1])).cls == "within-bound"
    assert compare(np.array([1, 3], np.uint8), V([1, 2], [0, 0])).cls == "off-by-one"
    c = compare(np.array([1, 9], np.uint8), V([1, 2], [0, 1]))
    assert c.cls == "mismatch" and c.example["index"] == [1] and c.max_excess == 6
    assert compare(np.array([np.nan], np.float32), V([1.0], [0.1], "float32")).cls == "nan-inf"
    assert compare(np.array([np.nan], np.float32), V([np.nan], [0], "float32")).cls == "exact"
    assert compare(np.array([5], np.uint8), V([1], [np.inf])).cls == "within-bound"


def simple_q_case(x, scale=0.5, zp=10):
    return Case([NodeSpec("QuantizeLinear", ["x", "s", "z"], ["y"])],
                [TensorSpec("x", "float32", np.asarray(x, np.float32))],
                [TensorSpec("s", "float32", np.float32(scale)), TensorSpec("z", "uint8", np.array(zp, np.uint8))],
                ["y"], {"pattern": "unit"})


def test_ort_runs_all_levels_and_matches_reference():
    case = simple_q_case([0.0, 1.0, 1.2, -3.3, 1000.0])
    res = run_all(case)
    assert set(res) == {"disable", "extended", "all"}
    assert all(r.error is None for r in res.values())
    v = judge(case, res)
    assert v.cls == "exact", v.to_json()
    assert v.signature.startswith("exact|")


def test_judge_detects_injected_mismatch():
    case = simple_q_case([0.0, 1.0, 1.2, -3.3, 1000.0])
    res = run_all(case)
    res["all"].outputs["y"] = res["all"].outputs["y"].copy()
    res["all"].outputs["y"][2] += 3
    v = judge(case, res)
    assert v.cls == "mismatch" and v.levels == ["all"]
    assert v.detail["example"]["index"] == [2]


def test_judge_reports_rejected_model():
    # per-row a_zero_point is valid ONNX but ORT's MatMulInteger rejects it
    a = TensorSpec("a", "uint8", np.ones((2, 2), np.uint8))
    case = Case([NodeSpec("MatMulInteger", ["a", "b", "az"], ["y"])], [a],
                [TensorSpec("b", "uint8", np.ones((2, 2), np.uint8)), TensorSpec("az", "uint8", np.array([1, 2], np.uint8))],
                ["y"], {"pattern": "unit"})
    v = judge(case)
    # Either ORT rejects it (known today) or computes it exactly (fixed upstream): never a silent mismatch.
    assert v.cls in ("rejects-valid-model", "exact")


def test_scale_regime():
    assert scale_regime(simple_q_case([1.0], scale=0.5)) == "normal"
    assert scale_regime(simple_q_case([1.0], scale=1e-40)) == "extreme-scale"
