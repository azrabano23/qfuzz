# ORT's DoubleQDQPairsRemover changes the result of Q -> DQ -> Q -> DQ (needs onnx, onnxruntime, numpy only)
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP
from onnx.reference import ReferenceEvaluator

def c(name, dt, v): return h.make_tensor(name, dt, [], [v])
nodes = [h.make_node("QuantizeLinear", ["x", "s1", "z1"], ["q1"]),
         h.make_node("DequantizeLinear", ["q1", "s1", "z1"], ["d1"]),
         h.make_node("QuantizeLinear", ["d1", "s2", "z2"], ["q2"]),
         h.make_node("DequantizeLinear", ["q2", "s2", "z2"], ["y"])]
g = h.make_graph(nodes, "double_qdq", [h.make_tensor_value_info("x", TP.FLOAT, [2])],
                 [h.make_tensor_value_info("y", TP.FLOAT, [2])],
                 [c("s1", TP.FLOAT, 1.0), c("z1", TP.INT16, 0), c("s2", TP.FLOAT, 0.001), c("z2", TP.INT16, 0)])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
onnx.checker.check_model(m, full_check=True)
x = np.array([-6.504, 2.4], np.float32)

# Spec: each Q rounds x/scale to nearest even, each DQ multiplies back -> the s1=1 stage snaps to integers.
q1 = np.clip(np.rint(x / np.float32(1.0)), -32768, 32767)
q2 = np.clip(np.rint(q1 / np.float32(0.001)), -32768, 32767)
expected = (q2 * np.float32(0.001)).astype(np.float32)
reference = ReferenceEvaluator(m).run(None, {"x": x})[0]

def run(level, **cfg):
    so = ort.SessionOptions(); so.graph_optimization_level = level
    for k, v in cfg.items(): so.add_session_config_entry(k, v)
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

L = ort.GraphOptimizationLevel
print("onnxruntime", ort.__version__, "onnx", onnx.__version__)
print("expected (spec formula)        ", expected)
print("onnx.reference                 ", reference)
print("ORT_DISABLE_ALL                ", run(L.ORT_DISABLE_ALL))
print("ORT_ENABLE_BASIC               ", run(L.ORT_ENABLE_BASIC))
print("ORT_ENABLE_ALL                 ", run(L.ORT_ENABLE_ALL))
print("ORT_ENABLE_ALL + disable remover", run(L.ORT_ENABLE_ALL, **{"session.disable_double_qdq_remover": "1"}))
assert np.array_equal(reference, expected)
assert np.array_equal(run(L.ORT_ENABLE_ALL), expected), "optimized graph differs from the ONNX spec"
