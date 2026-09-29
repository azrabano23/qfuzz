# Blocked (and 4-bit per-axis) QuantizeLinear saturates +inf and |x/scale| >= 2^31 to the wrong end of the range

> Paste-ready report for https://github.com/microsoft/onnxruntime/issues/new?template=01-bug_report.yml
> **Title:** `QuantizeLinear with block_size (and int4/uint4 per-axis): +inf and |x/scale| >= 2^31 saturate to the type minimum instead of the maximum`
> Suggested labels: `core runtime`, `quantization`. Not filed yet.

### Describe the issue

The spec (`onnx.defs.get_schema("QuantizeLinear", 21).doc`) says:

> The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.
> Saturation is done according to:
> - uint16: [0, 65535]
> - int16: [-32768, 32767]
> - uint8: [0, 255]
> - int8: [-128, 127]
> - uint4: [0, 15]
> - int4: [-8, 7]

So `x = +inf` or `x = 3e9` with scale 1 must give the type **maximum**. ORT's CPU kernel returns the type
**minimum** for:
- blocked quantization (`block_size > 0`) along an axis that is not the innermost one, for int8, uint8, int16,
  uint16, int4 and uint4;
- per-axis quantization of the innermost axis, for int4 and uint4.

Per-tensor quantization is correct for the same values, and so is blocked quantization along the innermost axis for
8/16-bit types. Values with |x/scale| < 2^31 are always correct. With a negative zero point the wrap-around shows up
as a sign flip in the other direction too: int4 blocked, `zp = -7`, `x = -3e38` gives **+7** instead of -8.

Measured with `x = [+inf, 3e9, 2e9, -inf]`, scale 1, zero point 0, for each layout:

| type | blocked, axis 0 of [2,2] | blocked, innermost axis | per-axis, innermost | per-tensor | spec |
|---|---|---|---|---|---|
| uint8 | **0 0** 255 0 | 255 255 255 0 | 255 255 255 0 | 255 255 255 0 | 255 255 255 0 |
| int16 | **-32768 -32768** 32767 -32768 | ok | ok | ok | 32767 32767 32767 -32768 |
| int4 | **-8 -8** 7 -8 | 7 7 7 -8 | **-8 -8** 7 -8 | 7 7 7 -8 | 7 7 7 -8 |
| uint4 | **0 0** 15 0 | 15 15 15 0 | **0 0** 15 0 | 15 15 15 0 | 15 15 15 0 |

**Cause (from reading the source).** The failing paths convert to `int32` *before* clamping:
[`onnxruntime/core/util/qmath.h` @ v1.30.0](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/util/qmath.h)

```cpp
// BlockedQuantizeLinear<float, TOut, 0>::opNotLastAxis, qmath.h#L489
auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx] / sc)) + zp, low, high);
```

`static_cast<int32_t>` of a float outside the int32 range is undefined behaviour. On x86 (`cvttss2si`) it yields
`INT32_MIN`, which then clamps to `low`. The same pattern appears in:
- `BlockedQuantizeLinear<float, TOut, 2>` (int4/uint4, both axes), L675-L761;
- the MLFloat16 variants, L578, L625, L821-L921, L1064, L1110;
- the 2-bit specialisation `BlockedQuantizeLinear<float, TOut, 3>`, L970, L1016 (not tested);
- the partial-byte head/tail of `DEFINE_PAR_QUANT_LINEAR_STD_4BIT`, L167, L182, and its 2-bit counterpart, L257-L308.
  The int4/uint4 `ComputeLoop` calls this once per channel, so per-axis quantization of the innermost axis sends
  every element through the scalar head/tail path;
- `ParQuantizeLinearStd(const MLFloat16*...)`, L341.

The per-tensor float path goes through `MlasQuantizeLinear`, which clamps in float first, and that is why it is
correct.

**Suggested fix:** clamp in float before converting, for example
`std::clamp(std::nearbyint(x / sc) + float(zp), float(low), float(high))`, then cast. NaN handling can stay as it is,
since the spec leaves NaN undefined for integer outputs.

**Note on `onnx.reference`.** `onnx.reference.ReferenceEvaluator` (onnx 1.23.0 and `main`) has the same defect:
`op_quantize_linear.py` does `np.rint(x).astype(np.int32)` before `np.clip`. It therefore cannot serve as the oracle
here, and the script prints its output for information only. The expected values come from the spec formula above,
and from ORT's own per-tensor kernel, which agrees with it. (A separate report for onnx/onnx is drafted next to this
file.)

### To reproduce

The script needs only `onnx`, `onnxruntime` and `numpy`. It builds the models inline, and the `DequantizeLinear`
(scale 1, zp 0) converts back to float for printing. It exits with an `AssertionError` while the bug is present.

```python
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
```

Output on onnxruntime 1.30.0 (PyPI wheel, `git-commit-id=f2c39fe`):

```
onnxruntime 1.30.0 onnx 1.23.0
x = [   inf 3.e+09 2.e+09   -inf]  scale = 1, zero_point = 0
INT8  expected [ 127.  127.  127. -128.] | ORT blocked: DISABLE_ALL [-128. -128.  127. -128.] ENABLE_ALL [-128. -128.  127. -128.] | ORT per-tensor [ 127.  127.  127. -128.] | onnx.reference blocked [-128. -128.  127. -128.]
UINT8 expected [255. 255. 255.   0.] | ORT blocked: DISABLE_ALL [  0.   0. 255.   0.] ENABLE_ALL [  0.   0. 255.   0.] | ORT per-tensor [255. 255. 255.   0.] | onnx.reference blocked [  0.   0. 255.   0.]
INT4  expected [ 7.  7.  7. -8.] | ORT blocked: DISABLE_ALL [-8. -8.  7. -8.] ENABLE_ALL [-8. -8.  7. -8.] | ORT per-tensor [ 7.  7.  7. -8.] | onnx.reference blocked [-8. -8.  7. -8.]
Traceback (most recent call last):
  File "rc4_blocked_quantize_saturation.py", line 34, in <module>
    assert ok, "blocked QuantizeLinear does not saturate as the ONNX spec requires"
           ^^
AssertionError: blocked QuantizeLinear does not saturate as the ONNX spec requires
# (exit code 1)
```

### Urgency

Medium-low. Blocked int4/int8 `QuantizeLinear` is used for weight-only LLM quantization and on-the-fly activation
quantization. An activation that overflowed to `+inf` upstream (fp16/fp32 overflow) becomes the most negative code
instead of saturating, which silently flips its sign. Ordinary scales are enough to trigger it once an `inf` appears.

### Platform

Linux

### OS Version

Ubuntu 24.04.4 LTS (kernel 6.18), x86-64

### ONNX Runtime Installation

Released Package

### ONNX Runtime Version or Commit ID

1.30.0 (`git-commit-id=f2c39fe`). `quantize_linear.cc` is byte-identical on `main` as of 2026-09-29. `qmath.h` on `main`
differs only by an early return in the 2-bit helper, and every conversion site listed above is unchanged.

### ONNX Runtime API

Python

### Architecture

X64

### Execution Provider

Default CPU

### Execution Provider Library Version

_No response_

---

**Related issues (searched 2026-09-29 for "QuantizeLinear block_size saturation", "int4 wrong saturation", "QuantizeLinear inf"; none is a duplicate):**
- #3442 (closed, old): QuantizeLinear int8 saturated to [-127, 127]. The symptom is similar, but the cause is different.
- PR #32452: fixed float16 QuantizeLinear *rounding* (it added `std::nearbyint` before `static_cast<int32_t>`) in
  the same file. It did not address out-of-range conversion.
- #32730 (open): WebGPU EP DequantizeLinear with a 1-D blocked input. That is a different EP and operator.
