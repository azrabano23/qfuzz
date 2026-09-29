"""Execute a Case with Apache TVM (Relax ONNX frontend, ``llvm`` CPU target).

This is an optional second backend. ``import qfuzz.tvm_runner`` works without
TVM, and :func:`available` reports whether ``tvm`` can be imported. The result is
an :class:`~qfuzz.runner.OrtResult` with ``level="tvm"``, so
:func:`qfuzz.oracle.judge` can compare it against the exact reference like any
ORT optimization level.

Pipeline (TVM >= 0.20, where Relay was removed)::

    from_onnx(model, keep_params_in_input=False)   # stage "import"
    -> DecomposeOpsForInference -> LegalizeOps
    -> tvm.compile(mod, target="llvm")              # stage "build"
    -> relax.VirtualMachine(...)["main"](*inputs)   # stage "run"

``stage`` on an error tells an importer limitation apart from a compiler or runtime
failure. :func:`classify_error` then separates *documented* unsupported features
(the importer raises "not supported" / OpNotImplemented) from everything else.
"""

from __future__ import annotations

import re

import numpy as np

from .case import DTYPES, FLOAT8_TYPES, OPAQUE_TYPES, Case
from .runner import OrtResult

LEVEL = "tvm"
TARGET = "llvm"

# ONNX ops the Relax ONNX importer (TVM 0.27) converts. QLinearMatMul, QLinearConv,
# ConvInteger and com.microsoft.QLinearAdd/Mul are missing, and blocked
# (block_size>0) DequantizeLinear and float8 saturate=0 QuantizeLinear are rejected
# explicitly.
SUPPORTED_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulInteger", "MatMul",
                 "Conv", "Add", "Mul", "Relu", "Cast", "Identity"}
# generator patterns whose op set is inside SUPPORTED_OPS (used by the TVM campaign)
PATTERNS = ["quantize", "dequantize", "qdq", "dq_matmul_q", "matmul_integer", "dq_conv_q", "dq_binary_q",
            "dq_relu_q", "dynamic_quantize", "dyn_matmul", "dq_matmul_nbits", "chain"]

_tvm = None


def available() -> bool:
    try:
        _load()
    except Exception:  # noqa: BLE001
        return False
    return True


def version() -> str | None:
    try:
        return _load()[0].__version__
    except Exception:  # noqa: BLE001
        return None


def _load():
    global _tvm
    if _tvm is None:
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        _tvm = (tvm, relax, from_onnx)
    return _tvm


def _short(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ")
    s = re.sub(r"/[^ ]*/site-packages/", "", s)
    s = re.sub(r"0x[0-9a-f]+", "0x?", s)
    return s[:400]


def _to_numpy(t, dtype: str | None) -> np.ndarray:
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if dtype in FLOAT8_TYPES:
        return np.ascontiguousarray(a).view(np.uint8)
    if dtype is not None and dtype in DTYPES and a.dtype != DTYPES[dtype][1]:
        return a.astype(DTYPES[dtype][1])
    return a


def run(case: Case) -> OrtResult:
    tvm, relax, from_onnx = _load()
    res = OrtResult(LEVEL)
    for t in case.inputs:
        if t.dtype in OPAQUE_TYPES or t.dtype in FLOAT8_TYPES:
            res.error, res.stage = f"qfuzz: cannot feed {t.dtype} graph inputs to TVM", "import"
            return res
    model = case.to_model()
    try:
        mod = from_onnx(model, keep_params_in_input=False)
    except Exception as e:  # noqa: BLE001
        res.error, res.stage = _short(e), "import"
        return res
    try:
        mod = relax.transform.DecomposeOpsForInference()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        ex = tvm.compile(mod, target=TARGET)
        vm = relax.VirtualMachine(ex, tvm.cpu())
    except Exception as e:  # noqa: BLE001
        res.error, res.stage = _short(e), "build"
        return res
    try:
        mk = tvm.runtime.tensor if hasattr(tvm.runtime, "tensor") else tvm.nd.array
        # np.ascontiguousarray would turn 0-d inputs into shape (1,); copy() keeps the rank
        args = [mk(np.array(t.data, copy=True, order="C"), tvm.cpu()) for t in case.inputs]
        out = vm["main"](*args)
        outs = list(out) if isinstance(out, (list, tuple)) or type(out).__name__ in ("Array", "Tuple") else [out]
        types = case.dtypes()
        res.outputs = {o: _to_numpy(v, types.get(o)) for o, v in zip(case.outputs, outs)}
        if len(res.outputs) != len(case.outputs):
            raise RuntimeError(f"expected {len(case.outputs)} outputs, got {len(outs)}")
    except Exception as e:  # noqa: BLE001
        res.outputs, res.error, res.stage = None, _short(e), "run"
    return res


_UNSUPPORTED = re.compile(r"not supported|OpNotImplemented|is not implemented|unsupported|should be one of", re.I)


def classify_error(res: OrtResult) -> str:
    """'tvm-unsupported' for an explicit importer refusal, else 'tvm-error-<stage>'."""
    if res.stage == "import" and _UNSUPPORTED.search(res.error or ""):
        return "tvm-unsupported"
    return f"tvm-error-{res.stage}"


def judge(case: Case, with_ort: bool = True):
    """Verdict for TVM against the exact reference (and, as a statistic, against ORT_DISABLE_ALL).

    TVM errors do not become findings. They are classified as 'tvm-unsupported'
    (explicit importer refusal) or 'tvm-error-<stage>' and triaged by hand.
    """
    from .oracle import Verdict, _norm_err, judge as _judge
    r = run(case)
    ops = "+".join(case.op_types())
    pattern = case.meta.get("pattern", "?")
    if r.error:
        cls = classify_error(r)
        return Verdict(cls, [LEVEL], None, None, f"{cls}|{ops}|{pattern}|{_norm_err(r.error)}",
                       {"error": r.error, "stage": r.stage})
    v = _judge(case, {LEVEL: r})
    if with_ort:
        from .runner import run_level
        o = run_level(case, "disable")
        if o.error is None:
            same = all(np.array_equal(np.asarray(o.outputs[k]), np.asarray(r.outputs[k]),
                                      equal_nan=np.asarray(r.outputs[k]).dtype.kind == "f")
                       for k in case.outputs)
            v.stats["tvm_eq_ort_disable"] = bool(same)
    return v
