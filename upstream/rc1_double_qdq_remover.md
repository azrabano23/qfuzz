# DoubleQDQPairsRemover changes the result of Q -> DQ -> Q -> DQ chains when the two pairs have different scales or zero points

> Paste-ready report for https://github.com/microsoft/onnxruntime/issues/new?template=01-bug_report.yml
> **Title:** `DoubleQDQPairsRemover changes numeric results of Q->DQ->Q->DQ chains (optimized graph != ORT_DISABLE_ALL != ONNX spec)`
> Suggested labels: `core runtime`, `quantization`. Not filed yet.

### Describe the issue

With graph optimizations at `ORT_ENABLE_BASIC` or higher (so also with the default `ORT_ENABLE_ALL`), the Level-1
transformer `DoubleQDQPairsRemover` rewrites

```
x -> QuantizeLinear(s1, z1) -> DequantizeLinear(s1, z1) -> QuantizeLinear(s2, z2) -> DequantizeLinear(s2, z2)
```

into a single `QuantizeLinear -> DequantizeLinear` pair. When `(s1, z1) != (s2, z2)` the new pair gets a *recomputed*
scale and zero point
([`FindNewZeroPointAndScale`, double_qdq_pairs_remover.cc#L119-L128 @ v1.30.0](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/optimizer/double_qdq_pairs_remover.cc#L119-L128)):

```cpp
float real_min1 = gsl::narrow_cast<float>(q_min - zero_point_1) * scale_1;
...
const float real_min = std::max(real_min1, real_min2);
const float real_max = std::min(real_max1, real_max2);
new_scale = (real_max - real_min) / gsl::narrow_cast<float>(q_max - q_min);
new_zero_point = gsl::narrow_cast<T>(std::round(gsl::narrow_cast<float>(q_min) - real_min / new_scale));
```

That is the intersection of the two representable ranges spread over the full integer range. This grid is
usually neither `s1` nor `s2`, so the rewritten graph computes a different function. In particular the rounding of
the coarser stage is lost: in the repro below (int16, `s1 = 1`, `s2 = 0.001`), the first pair should snap `x` to an
integer, but after the rewrite only the fine `s2` grid is left and `-6.504` comes back unchanged instead of `-7`.

**Why the unoptimized result is the correct one.** The ONNX spec defines the graph as the composition of the four
operators. It gives no licence to merge them. From `onnx.defs.get_schema("QuantizeLinear", 21).doc`:

> The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`. [...] For `(x / y_scale)`, it rounds to the nearest even.

and from `DequantizeLinear`:

> The dequantization formula is `y = (x - x_zero_point) * x_scale`.

So for `x = -6.504`: Q1 = round(-6.504 / 1) = -7, DQ1 = -7.0, Q2 = round(-7.0 / 0.001) = -7000,
DQ2 = -7000 * float32(0.001) = **-7.0000005**. `onnx.reference.ReferenceEvaluator` and `ORT_DISABLE_ALL` both return
that value. `ORT_ENABLE_BASIC` and `ORT_ENABLE_ALL` return **-6.504**. The same happens with `x = 2.4` (spec 2.0,
optimized 2.4). As an independent check, Apache TVM 0.27 (Relax ONNX frontend, `llvm` target) also returns
`[-7.0000005, 2.0]` for this model.

Other instances found by differential fuzzing (qfuzz, 150k random cases, 653 hits with this root cause, in all of
which `ORT_DISABLE_ALL` matches the spec):
- uint16, `s1 = s2 = 1`, `z1 = 0`, `z2 = 30458`, `x = 20000`: spec 20000.0, optimized **19999.805**. The ranges
  [0, 65535] and [-30458, 35077] intersect in [0, 35077], so `new_scale = 35077/65535`, and integers are no longer
  exactly representable.
- int8, `s1 = 1e-4`, `s2 = 1`, followed by Relu, `x = 0.01`: spec 0.0, optimized 0.01.
- With very large scales (about 3e34) the range computation overflows float32. `new_scale` becomes inf, and an
  input of 0 comes out as NaN.

**Workaround:** `so.add_session_config_entry("session.disable_double_qdq_remover", "1")` restores the correct result
(shown in the repro).

**Suggested fix:** only rewrite when it is exactly semantics-preserving. That covers the existing equal-(scale, zp)
fast path, and the case where Q2(DQ1(q)) is the identity on every representable q1 and every x (for example
`z1 == z2` and `s2` divides `s1` with nested ranges). Where a merge is kept, compute the new parameters in double
precision so the range product cannot overflow.

### To reproduce

The script needs only `onnx`, `onnxruntime` and `numpy`. It builds the model inline and exits with an
`AssertionError` while the bug is present.

```python
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
```

Output on onnxruntime 1.30.0 (PyPI wheel, `git-commit-id=f2c39fe`):

```
onnxruntime 1.30.0 onnx 1.23.0
expected (spec formula)         [-7.0000005  2.       ]
onnx.reference                  [-7.0000005  2.       ]
ORT_DISABLE_ALL                 [-7.0000005  2.       ]
ORT_ENABLE_BASIC                [-6.504  2.4  ]
ORT_ENABLE_ALL                  [-6.504  2.4  ]
ORT_ENABLE_ALL + disable remover [-7.0000005  2.       ]
Traceback (most recent call last):
  File "rc1_double_qdq_remover.py", line 38, in <module>
    assert np.array_equal(run(L.ORT_ENABLE_ALL), expected), "optimized graph differs from the ONNX spec"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: optimized graph differs from the ONNX spec
# (exit code 1)
```

Dumping the optimized model (`so.optimized_model_filepath`) shows a single `QuantizeLinear -> DequantizeLinear`
with initializers `DoubleQDQRemoved_s2 = 0.001`, `DoubleQDQRemoved_z1 = 0`, `DoubleQDQRemoved_z2 = 0`.

### Urgency

Not blocking. It is a silent wrong result from an optimization that is on by default (Level 1), and it affects any
model with back-to-back QDQ pairs whose parameters differ. Quantization tools emit such pairs when requantizing
between layers with different calibration ranges. The workaround is the session option above.

### Platform

Linux

### OS Version

Ubuntu 24.04.4 LTS (kernel 6.18), x86-64

### ONNX Runtime Installation

Released Package

### ONNX Runtime Version or Commit ID

1.30.0 (`git-commit-id=f2c39fe`). The relevant source (`double_qdq_pairs_remover.cc`, `graph_transformer_utils.cc`) is
byte-identical on `main` as of 2026-09-29.

### ONNX Runtime API

Python

### Architecture

X64

### Execution Provider

Default CPU

### Execution Provider Library Version

_No response_

---

**Related issues (searched 2026-09-29 for "DoubleQDQPairsRemover", "double QDQ", "Q DQ Q DQ scale"; none is a duplicate):**
- #32132 (open): fused QLinearAdd/QLinearConv/QLinearAveragePool off by k quantization steps. This is a different
  transformer (QDQSelectorActionTransformer), and `ORT_ENABLE_BASIC` is correct there. Here BASIC is already wrong.
- #28030 (closed): a double QDQ pair crashes QLinearAveragePool. It is the same transformer, but it reports a crash, not
  wrong numbers.
- #21319 (closed): QDQ removal around Resize changes values. The symptom is similar, but the transformer is different.
