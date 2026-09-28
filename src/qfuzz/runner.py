"""Execute a Case under ONNX Runtime at several graph-optimization levels."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from .case import Case

LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

ort.set_default_logger_severity(4)


@dataclass
class OrtResult:
    level: str
    outputs: dict[str, np.ndarray] | None = None
    error: str | None = None
    stage: str | None = None  # "load" or "run"
    fused_ops: list[str] = field(default_factory=list)


def session_options(level: str, optimized_path: str | None = None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = LEVELS[level]
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 4
    if optimized_path:
        so.optimized_model_filepath = optimized_path
    return so


def run_level(case: Case, level: str, model_bytes: bytes | None = None,
              record_fused: bool = False) -> OrtResult:
    if model_bytes is None:
        model_bytes = case.to_model().SerializeToString()
    res = OrtResult(level)
    opt_path = None
    if record_fused:
        fd, opt_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
    try:
        sess = ort.InferenceSession(model_bytes, session_options(level, opt_path),
                                    providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        res.error, res.stage = _short(e), "load"
        return res
    finally:
        pass
    if record_fused and opt_path:
        try:
            import onnx
            m = onnx.load(opt_path)
            res.fused_ops = [f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in m.graph.node]
        except Exception:  # noqa: BLE001
            pass
        finally:
            os.unlink(opt_path)
    try:
        outs = sess.run(case.outputs, case.feeds())
        res.outputs = dict(zip(case.outputs, outs))
    except Exception as e:  # noqa: BLE001
        res.error, res.stage = _short(e), "run"
    return res


def _short(e: Exception) -> str:
    s = str(e).replace("\n", " ")
    # strip absolute build paths to keep signatures stable across builds
    import re
    s = re.sub(r"/[^ ]*/onnxruntime/", "onnxruntime/", s)
    return s[:400]


def run_all(case: Case, levels=("disable", "extended", "all"), record_fused: bool = False) -> dict[str, OrtResult]:
    mb = case.to_model().SerializeToString()
    return {lv: run_level(case, lv, mb, record_fused=record_fused) for lv in levels}
