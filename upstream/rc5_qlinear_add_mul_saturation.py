# DQ -> Add/Mul -> Q fused into QLinearAdd/QLinearMul saturates to the WRONG end when the result exceeds int32
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP
from onnx.reference import ReferenceEvaluator

def model(op):  # standard ONNX ops only; ORT's QDQ fusion turns this into com.microsoft.QLinear<op>
    nodes = [h.make_node("DequantizeLinear", ["a", "sa", "z"], ["fa"]),
             h.make_node("DequantizeLinear", ["b", "sb", "z"], ["fb"]),
             h.make_node(op, ["fa", "fb"], ["fc"]),
             h.make_node("QuantizeLinear", ["fc", "sc", "z"], ["c"])]
    vi = lambda n: h.make_tensor_value_info(n, TP.UINT8, [4])
    g = h.make_graph(nodes, "qdq_" + op, [vi("a"), vi("b")], [vi("c")],
                     [h.make_tensor("sa", TP.FLOAT, [], [3000.0]), h.make_tensor("sb", TP.FLOAT, [], [1.0]),
                      h.make_tensor("sc", TP.FLOAT, [], [1e-4]), h.make_tensor("z", TP.UINT8, [], [0])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(m, full_check=True)
    return m

def run(m, feeds, level):
    so = ort.SessionOptions(); so.graph_optimization_level = level
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)[0]

print("onnxruntime", ort.__version__, "onnx", onnx.__version__)
feeds = {"a": np.array([0, 1, 60, 229], np.uint8), "b": np.array([0, 2, 1, 1], np.uint8)}
L, ok = ort.GraphOptimizationLevel, True
for op, f in [("Add", np.add), ("Mul", np.multiply)]:
    m = model(op)
    exact = f(feeds["a"] * 3000.0, feeds["b"] * 1.0) / 1e-4  # up to 6.9e9 > 2^31
    want = np.clip(np.rint(exact), 0, 255).astype(np.uint8)
    with np.errstate(invalid="ignore"):  # onnx.reference casts to int32 before clipping, see report
        ref = ReferenceEvaluator(m).run(None, feeds)[0]
    off, on = run(m, feeds, L.ORT_DISABLE_ALL), run(m, feeds, L.ORT_ENABLE_ALL)
    print(f"{op}: exact (a*sa {op} b*sb)/sc = {exact}")
    print(f"{op}: expected {want}  onnx.reference {ref}  ORT_DISABLE_ALL {off}  ORT_ENABLE_ALL {on}")
    assert np.array_equal(off, want)
    ok &= np.array_equal(on, want)
assert ok, "fused QLinearAdd/QLinearMul result differs from the ONNX spec (and from the unfused graph)"
