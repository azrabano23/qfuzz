# [onnx/onnx] Reference QuantizeLinear casts to int32 before clipping, so +inf and |x/scale| >= 2^31 saturate to the wrong end

> This one is for **onnx/onnx**, not onnxruntime: https://github.com/onnx/onnx/issues/new?template=bug.md
> **Title:** `Reference QuantizeLinear: np.rint(x).astype(np.int32) before np.clip makes +inf / |x| >= 2^31 saturate to qmin`
> It was found while writing the RC4 and RC5 onnxruntime reports: the reference could not serve as the oracle there. Not filed yet.

# Bug Report

### Is the issue related to model conversion?

No.

### Describe the bug

The spec (`onnx.defs.get_schema("QuantizeLinear").doc`) says:

> The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.
> Saturation is done according to: [...] uint8: [0, 255] [...] int8: [-128, 127]

`onnx/reference/ops/op_quantize_linear.py` (onnx 1.23.0, unchanged on `main` as of 2026-09-29) does:

```python
xi = np.rint(x).astype(np.int32)
if zero_point is not None:
    xi += zero_point
...
return (np.clip(xi, quant_range[0], quant_range[1]).astype(dtype),)
```

A NumPy float to int32 cast of +inf or of values >= 2^31 is undefined. On x86-64 it yields INT32_MIN and emits
`RuntimeWarning: invalid value encountered in cast`, so `+inf` and `3e9` quantize to the **minimum** (uint8 0,
int8 -128) instead of the maximum. The `+= zero_point` in int32 can also wrap. The same code serves per-tensor,
per-axis and blocked quantization. The float8 branch is not affected.

onnxruntime's per-tensor CPU kernel returns the spec value (255 / 127) for the same inputs.

### System information

- OS Platform and Distribution: Ubuntu 24.04.4 LTS, x86-64
- ONNX version: 1.23.0 (code identical on `main`)
- Python version: 3.11
- NumPy: 2.4.6

### Reproduction instructions

```python
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
```

Output:

```
onnx 1.23.0 numpy 2.4.6
UINT8 x=[    inf  3.e+09  2.e+09 -3.e+09    -inf]  expected [255 255 255   0   0]  onnx.reference [  0   0 255   0   0]
INT8  x=[    inf  3.e+09  2.e+09 -3.e+09    -inf]  expected [ 127  127  127 -128 -128]  onnx.reference [-128 -128  127 -128 -128]
INT16 x=[    inf  3.e+09  2.e+09 -3.e+09    -inf]  expected [ 32767  32767  32767 -32768 -32768]  onnx.reference [-32768 -32768  32767 -32768 -32768]
Traceback (most recent call last):
  File "onnx_reference_quantize_overflow.py", line 20, in <module>
    assert ok, "onnx.reference QuantizeLinear does not saturate out-of-int32-range inputs"
           ^^
AssertionError: onnx.reference QuantizeLinear does not saturate out-of-int32-range inputs
# (exit code 1)
```

### Expected behavior

Clip in floating point before the integer cast, for example
`np.clip(np.rint(x).astype(np.float64) + zero_point, qmin, qmax).astype(dtype)`.
This is a bug in the reference implementation and needs no spec change. NaN stays undefined.

### Notes

Related, not duplicates (searched 2026-09-29): onnx/onnx#7835 (closed; QLinearMatMul reference wrapped instead of
saturating), onnx/onnx#8227 (closed; QuantizeLinear reference -0.0 handling for low-precision floats).
