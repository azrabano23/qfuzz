# QuantizeLinear(float8, saturate=0): wrong result for values whose rounding lands just above FLT_MAX
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP
from onnx.reference import ReferenceEvaluator

def model(f8):
    nodes = [h.make_node("QuantizeLinear", ["x", "s", "z"], ["q"], saturate=0),
             h.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])]  # back to float32 for printing
    g = h.make_graph(nodes, "f8", [h.make_tensor_value_info("x", TP.FLOAT, [5])],
                     [h.make_tensor_value_info("y", TP.FLOAT, [5])],
                     [h.make_tensor("s", TP.FLOAT, [], [1.0]), h.make_tensor("z", f8, [], [0.0])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m, full_check=True)
    return m

def run(m, x, level):
    so = ort.SessionOptions(); so.graph_optimization_level = level
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

print("onnxruntime", ort.__version__, "onnx", onnx.__version__)
L, ok = ort.GraphOptimizationLevel, True
# E4M3FN: FLT_MAX=448, [x] > 448 -> NaN.  E5M2: FLT_MAX=57344, [x] > 57344 -> +Inf.
cases = {"FLOAT8E4M3FN": (TP.FLOAT8E4M3FN, [464, 479, 480, 495, 496], [448, np.nan, np.nan, np.nan, np.nan]),
         "FLOAT8E5M2": (TP.FLOAT8E5M2, [57344, 61439, 61440, 65000, 65536], [57344, 57344, np.inf, np.inf, np.inf])}
for name, (f8, xs, want) in cases.items():
    m, x, want = model(f8), np.array(xs, np.float32), np.array(want, np.float32)
    ref = ReferenceEvaluator(m).run(None, {"x": x})[0]
    off, on = run(m, x, L.ORT_DISABLE_ALL), run(m, x, L.ORT_ENABLE_ALL)
    print(f"{name}: x              ", x)
    print(f"{name}: expected (spec)", want)
    print(f"{name}: onnx.reference ", ref)
    print(f"{name}: ORT_DISABLE_ALL", off)
    print(f"{name}: ORT_ENABLE_ALL ", on)
    assert np.array_equal(ref, want, equal_nan=True)
    ok &= np.array_equal(off, want, equal_nan=True) and np.array_equal(on, want, equal_nan=True)
assert ok, "ORT float8 saturate=0 conversion differs from the ONNX spec"
