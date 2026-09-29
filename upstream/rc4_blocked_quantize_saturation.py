# Blocked QuantizeLinear saturates |x/scale| >= 2^31 (and +-inf) to the wrong end of the integer range
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP
from onnx.reference import ReferenceEvaluator

def model(qt, blocked):  # x: [2, 2]; blocked along axis 0 (NOT the innermost axis), or per-tensor
    kw, sh = (dict(axis=0, block_size=2), [1, 2]) if blocked else ({}, [])
    nodes = [h.make_node("QuantizeLinear", ["x", "s", "z"], ["q"], **kw),
             h.make_node("DequantizeLinear", ["q", "s", "z"], ["y"], **kw)]  # back to float for printing
    g = h.make_graph(nodes, "q", [h.make_tensor_value_info("x", TP.FLOAT, [2, 2])],
                     [h.make_tensor_value_info("y", TP.FLOAT, [2, 2])],
                     [h.make_tensor("s", TP.FLOAT, sh, [1.0] * len(sh or [1])), h.make_tensor("z", qt, sh, [0] * len(sh or [1]))])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m, full_check=True)
    return m

def run(m, x, level):
    so = ort.SessionOptions(); so.graph_optimization_level = level
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

print("onnxruntime", ort.__version__, "onnx", onnx.__version__)
x = np.array([[np.inf, 3e9], [2e9, -np.inf]], np.float32)  # 2e9 < 2^31 <= 3e9
print("x =", x.ravel(), " scale = 1, zero_point = 0")
L, ok = ort.GraphOptimizationLevel, True
for name, qt, lo, hi in [("INT8", TP.INT8, -128, 127), ("UINT8", TP.UINT8, 0, 255), ("INT4", TP.INT4, -8, 7)]:
    want = np.clip(x, lo, hi)  # spec: saturate(round(x / 1) + 0); DequantizeLinear(scale 1, zp 0) is exact
    blk, pt = model(qt, True), model(qt, False)
    with np.errstate(invalid="ignore"):
        ref = ReferenceEvaluator(blk).run(None, {"x": x})[0]
    off, on, per_tensor = run(blk, x, L.ORT_DISABLE_ALL), run(blk, x, L.ORT_ENABLE_ALL), run(pt, x, L.ORT_ENABLE_ALL)
    print(f"{name:5s} expected {want.ravel()} | ORT blocked: DISABLE_ALL {off.ravel()} ENABLE_ALL {on.ravel()}"
          f" | ORT per-tensor {per_tensor.ravel()} | onnx.reference blocked {ref.ravel()}")
    ok &= np.array_equal(off, want) and np.array_equal(on, want)
assert ok, "blocked QuantizeLinear does not saturate as the ONNX spec requires"
