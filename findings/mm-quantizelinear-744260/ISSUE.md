# DRAFT (not filed) -- QuantizeLinear (blocked / 4-bit) saturates +inf and |x/scale| >= 2^31 to the wrong end

**Component:** CPU EP `QuantizeLinear` (`onnxruntime/core/providers/cpu/quantization/quantize_linear.cc`)
**Version:** onnxruntime 1.30.0, Linux x86-64 (AVX-512 VNNI / AMX machine), model opset 21

## Describe the issue

`QuantizeLinear` must compute `saturate(round(x / y_scale) + y_zero_point)`. For blocked quantization
(`block_size > 0`) along a non-innermost axis, ORT returns the **minimum** of the output type instead of the
maximum when `x / y_scale >= 2^31`, including `x = +inf`:

| x | y_scale | zp | spec (uint8) | ORT |
|---|---|---|---|---|
| 1e9 | 1 | 0 | 255 | 255 |
| 2.2e9 | 1 | 0 | 255 | **0** |
| 3e38 | 1 | 0 | 255 | **0** |
| +inf | 1 | 0 | 255 | **0** |

int8 gives -128 instead of 127, int16 -32768 instead of 32767, and with a negative zero point a large *negative*
input comes out as the *maximum* (int4, x = -3e38, zp = -7 -> +7 instead of -8), i.e. the int32 wrap-around
of `INT32_MIN + zp` is visible. Per-tensor quantization and blocking along the innermost axis are correct; for
int4/uint4 per-axis quantization of the innermost axis shows the same problem. Values below 2^31 are always
correct, which suggests a float -> int32 conversion (x86 `cvtps2dq` returns `0x80000000` when out of range)
before clamping in the strided code path.

## Minimal repro

```python
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

x = np.zeros((1, 2, 2, 1), np.float32); x[0, 0, 0, 0] = np.inf
g = h.make_graph([h.make_node("QuantizeLinear", ["x", "s", "z"], ["y"], axis=1, block_size=2)], "g",
                 [h.make_tensor_value_info("x", TP.FLOAT, [1, 2, 2, 1])],
                 [h.make_tensor_value_info("y", TP.UINT8, [1, 2, 2, 1])],
                 initializer=[nh.from_array(np.ones((1, 1, 2, 1), np.float32), "s"),
                              nh.from_array(np.zeros((1, 1, 2, 1), np.uint8), "z")])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)]); m.ir_version = 10
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
print(s.run(None, {"x": x})[0].ravel())   # [0 0 0 0]  -- expected [255 0 0 0]
```

`repro.py` next to this file is the automatically minimized, self-checking version (x = 3e38).

## Why it matters

Blocked int4/int8 quantization is the format used for weight-only LLM quantization; `QuantizeLinear` with
blocking is also used for on-the-fly activation quantization. An `inf` produced upstream (fp16/fp32 overflow)
silently becomes the most *negative* code instead of saturating, flipping the sign of the value.

## Expected behavior

Clamp in floating point before converting to integer (as the per-tensor MLAS kernels already do), or saturate the
conversion.
