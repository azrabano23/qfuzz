# onnx.reference QuantizeLinear: |x/scale| >= 2^31 or +-inf saturates to the wrong end (needs onnx + numpy only)
import numpy as np, onnx
from onnx import helper as h, TensorProto as TP
from onnx.reference import ReferenceEvaluator

print("onnx", onnx.__version__, "numpy", np.__version__)
x = np.array([np.inf, 3e9, 2e9, -3e9, -np.inf], np.float32)  # 2e9 < 2^31 <= 3e9
ok = True
for name, qt, lo, hi in [("UINT8", TP.UINT8, 0, 255), ("INT8", TP.INT8, -128, 127), ("INT16", TP.INT16, -32768, 32767)]:
    g = h.make_graph([h.make_node("QuantizeLinear", ["x", "s", "z"], ["y"])], "q",
                     [h.make_tensor_value_info("x", TP.FLOAT, [5])], [h.make_tensor_value_info("y", qt, [5])],
                     [h.make_tensor("s", TP.FLOAT, [], [1.0]), h.make_tensor("z", qt, [], [0])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m, full_check=True)
    want = np.clip(x, lo, hi)  # spec: y = saturate((x / y_scale) + y_zero_point)
    with np.errstate(invalid="ignore"):
        got = ReferenceEvaluator(m).run(None, {"x": x})[0]
    print(f"{name:5s} x={x}  expected {want.astype(np.int64)}  onnx.reference {got.astype(np.int64)}")
    ok &= np.array_equal(got.astype(np.float64), want.astype(np.float64))
assert ok, "onnx.reference QuantizeLinear does not saturate out-of-int32-range inputs"
