# DRAFT (not filed) -- DoubleQDQPairsRemover changes results of Q -> DQ -> Q -> DQ chains with different scales

**Component:** graph optimizer, `onnxruntime/core/optimizer/double_qdq_pairs_remover.cc`
**Version:** onnxruntime 1.30.0 (CPU EP, Linux x86-64), onnx 1.23.0; model opset 21
**Found by:** differential fuzzing against an exact ONNX-spec reference (qfuzz), minimized automatically.

## Describe the issue

With `ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL`, a chain

```
x -> QuantizeLinear(s1, z1) -> DequantizeLinear(s1, z1) -> QuantizeLinear(s2, z2) -> DequantizeLinear(s2, z2)
```

is rewritten by `DoubleQDQPairsRemover` into a single `QuantizeLinear -> DequantizeLinear` pair whose scale is
recomputed in `FindNewZeroPointAndScale` as

```
new_scale = (min(real_max1, real_max2) - max(real_min1, real_min2)) / (qmax - qmin)
```

i.e. the *intersection of the two representable ranges* spread over the full integer range. That grid is in
general neither `s1` nor `s2`, so the optimized graph does not compute the same function as the original one. In
particular the rounding of the *coarser* of the two stages is lost.

ONNX defines the graph as the composition of the four operators, so the unoptimized result is the correct one
(and it is what `ORT_DISABLE_ALL` returns).

## Minimal repro (int16, realistic scales)

x = -6.504, s1 = 1.0, z1 = 0, s2 = 0.001, z2 = 0:

| step | spec value |
|---|---|
| Q1: round(-6.504 / 1) | -7 |
| DQ1 | -7.0 |
| Q2: round(-7.0 / 0.001) | -7000 |
| DQ2 | **-7.0** (-7.0000005 in float32, since float32(0.001) != 0.001) |

ORT at `ORT_ENABLE_EXTENDED` / `ORT_ENABLE_ALL` returns **-6.504** (the first, coarse quantization step has
vanished: the intersection of [-32768, 32767] and [-32.768, 32.767] is the second range, so the new pair is just
Q2/DQ2). `ORT_DISABLE_ALL` returns -7.0.

```python
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP

def c(name, dt, v): return h.make_tensor(name, dt, [], [v])
nodes = [h.make_node("QuantizeLinear", ["x", "s1", "z1"], ["q1"]),
         h.make_node("DequantizeLinear", ["q1", "s1", "z1"], ["d1"]),
         h.make_node("QuantizeLinear", ["d1", "s2", "z2"], ["q2"]),
         h.make_node("DequantizeLinear", ["q2", "s2", "z2"], ["y"])]
g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", TP.FLOAT, [1])],
                 [h.make_tensor_value_info("y", TP.FLOAT, [1])],
                 initializer=[c("s1", TP.FLOAT, 1.0), c("z1", TP.INT16, 0),
                              c("s2", TP.FLOAT, 0.001), c("z2", TP.INT16, 0)])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)]); m.ir_version = 10
x = np.array([-6.504], np.float32)
for lvl in [ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED]:
    so = ort.SessionOptions(); so.graph_optimization_level = lvl
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    print(lvl, s.run(None, {"x": x})[0])
# ORT_DISABLE_ALL      [-7.0000005]   (= -7000 * float32(0.001))
# ORT_ENABLE_EXTENDED  [-6.504]
```

(The generated, self-checking version of this repro is `repro.py` next to this file.)

## Other instances found by the same fuzzer

* uint16, x = 20000, s1 = s2 = 1, z1 = 0, z2 = 30458 -> spec 20000.0, optimized 19999.8047 (the ranges
  [0, 65535] and [-30458, 35077] intersect in [0, 35077], so the new scale is 35077/65535 != 1 and integers
  stop being representable).
* int8, Relu after the chain, x = 0.01, s1 = 1e-4, s2 = 1 -> spec 0.0, optimized 0.01.
* With large scales (s ~ 3e34) the range product overflows float32, `new_scale` becomes inf and an input of 0
  produces NaN.

653 of 150,000 random cases hit this root cause (435 + 133 with ordinary scales, 85 with extreme scales); in all of them `ORT_DISABLE_ALL` is correct.

## Expected behavior

Either skip the rewrite unless it is exactly semantics-preserving (e.g. identical `(scale, zero_point)` for both
pairs -- the case the existing `scale_1 == scale_2 && zp_1 == zp_2` fast path already handles), or only fold when
one grid is an integer refinement of the other and the ranges nest -- and compute the new parameters in double
precision so the range product cannot overflow.

## Notes

Searched existing issues for `DoubleQDQPairsRemover` / "double QDQ" -- related but distinct: #32132 (fused
QLinear* ops off by k steps), #28030 (double QDQ crash in QLinearAveragePool, closed).
