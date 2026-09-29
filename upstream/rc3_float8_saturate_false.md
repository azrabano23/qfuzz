# float8 conversion with saturate=0 is wrong just above FLT_MAX: E4M3FN returns 448 instead of NaN, E5M2 returns NaN instead of Inf

> Paste-ready report for https://github.com/microsoft/onnxruntime/issues/new?template=01-bug_report.yml
> **Title:** `QuantizeLinear/Cast to float8 with saturate=0: E4M3FN maps [480, 496) to 448 (should be NaN), E5M2 maps [61440, 65536) to NaN (should be Inf)`
> Suggested labels: `core runtime`. Not filed yet.

### Describe the issue

The ONNX spec defines the float8 conversion with `saturate=0` by a table (quoted from
`onnx.defs.get_schema("Cast").doc`, which `QuantizeLinear`'s `saturate` attribute refers to: *"All cases are fully
described in two tables inserted in the operator description"*):

> `[x]` means the value rounded to the target mantissa width.
>
> | x                 | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
> | ----------------- | ------ | -------- | ---- | -------- |
> | Inf               | NaN    | NaN      | Inf  | NaN      |
> | \[x\] > FLT_MAX   | NaN    | NaN      | Inf  | NaN      |
> | \[x\] \< -FLT_MAX | NaN    | NaN      | -Inf | NaN      |
> | else              | RNE    | RNE      | RNE  | RNE      |

The CPU implementation gets two bands wrong. This applies to both `QuantizeLinear` (scale 1, zero point 0) and `Cast`:

| type | x | `[x]` (RNE to target mantissa) | spec | `onnx.reference` | ORT 1.30 CPU |
|---|---|---|---|---|---|
| E4M3FN (FLT_MAX 448) | 464 | 448 (tie, to even) | 448 | 448 | 448 |
| E4M3FN | 479 | 480 | NaN | NaN | NaN |
| E4M3FN | **480 ... 495** | 480 | NaN | NaN | **448** |
| E4M3FN | 496 | 512 | NaN | NaN | NaN |
| E5M2 (FLT_MAX 57344) | 61439 | 57344 | 57344 | 57344 | 57344 |
| E5M2 | **61440 ... 65535** | 65536 | +Inf | +Inf | **NaN** |
| E5M2 | 65536 | 65536 | +Inf | +Inf | +Inf |

The E4M3FN result is not even monotonic: 479 gives NaN, 480 gives 448, 496 gives NaN. Negative inputs are mirrored.
The same results appear for per-tensor, per-axis and blocked `QuantizeLinear` and for `Cast(to=FLOAT8E4M3FN|FLOAT8E5M2, saturate=0)`.
The saturate=1 path is correct.

**Cause (from reading the source).** Both bugs are in the non-CUDA branch of
[`include/onnxruntime/core/common/float8.h` @ v1.30.0](https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/common/float8.h):
- `Float8E4M3FN(float v, bool saturate)`, [line 101](https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/common/float8.h#L101).
  When the truncated value hits the NaN encoding `0x7F` (x in [480, 512)), `val &= 0xFE` turns it into 448 even
  when `saturate == false`. For x in [480, 496) the rounding bit is 0, so nothing corrects it afterwards.
- `Float8E5M2(float v, bool saturate)`, rounding branch at [line 431](https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/common/float8.h#L431).
  When rounding up from `0x7B` (57344) overflows, the code does `val |= 0x7C`. That ORs the infinity pattern into
  the existing mantissa bits `0x7B` and gives `0x7F`, which is NaN. It should *set* the magnitude to `0x7C`:
  `val = (val & 0x80) | 0x7C`.

The CUDA branch (`__nv_cvt_float_to_fp8(..., __NV_NOSAT, ...)`) was not tested.

### To reproduce

The script needs only `onnx`, `onnxruntime` and `numpy`. It builds the models inline, and the `DequantizeLinear`
converts back to float32 for printing (DQ of these codes is exact). It exits with an `AssertionError` while the bug
is present.

```python
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
```

Output on onnxruntime 1.30.0 (PyPI wheel, `git-commit-id=f2c39fe`):

```
onnxruntime 1.30.0 onnx 1.23.0
FLOAT8E4M3FN: x               [464. 479. 480. 495. 496.]
FLOAT8E4M3FN: expected (spec) [448.  nan  nan  nan  nan]
FLOAT8E4M3FN: onnx.reference  [448.  nan  nan  nan  nan]
FLOAT8E4M3FN: ORT_DISABLE_ALL [448.  nan 448. 448.  nan]
FLOAT8E4M3FN: ORT_ENABLE_ALL  [448.  nan 448. 448.  nan]
FLOAT8E5M2: x               [57344. 61439. 61440. 65000. 65536.]
FLOAT8E5M2: expected (spec) [57344. 57344.    inf    inf    inf]
FLOAT8E5M2: onnx.reference  [57344. 57344.    inf    inf    inf]
FLOAT8E5M2: ORT_DISABLE_ALL [57344. 57344.    nan    nan    inf]
FLOAT8E5M2: ORT_ENABLE_ALL  [57344. 57344.    nan    nan    inf]
Traceback (most recent call last):
  File "rc3_float8_saturate_false.py", line 36, in <module>
    assert ok, "ORT float8 saturate=0 conversion differs from the ONNX spec"
           ^^
AssertionError: ORT float8 saturate=0 conversion differs from the ONNX spec
# (exit code 1)
```

The same values come from `Cast(x, to=FLOAT8E4M3FN, saturate=0)`. Its raw output bytes for x = 480 and 495 are
`0x7E` (448) instead of `0x7F` (NaN).

### Urgency

Low. `saturate=0` is not the default. When it is used, however, it is used precisely to detect overflow, and a value
that overflowed silently becomes a finite 448 (E4M3FN), or +/-Inf becomes NaN (E5M2).

### Platform

Linux

### OS Version

Ubuntu 24.04.4 LTS (kernel 6.18), x86-64

### ONNX Runtime Installation

Released Package

### ONNX Runtime Version or Commit ID

1.30.0 (`git-commit-id=f2c39fe`). `include/onnxruntime/core/common/float8.h` is byte-identical on `main` as of 2026-09-29.

### ONNX Runtime API

Python

### Architecture

X64

### Execution Provider

Default CPU

### Execution Provider Library Version

_No response_

---

**Related issues (searched 2026-09-29 for "float8 saturate", "E4M3FN NaN 448", "E5M2 NaN instead of infinity"; none is a duplicate):**
- #16938 (open): "Float8_e4m3 results are off-by-one". That is a different rounding discrepancy, not the overflow band.
- #32719 (open): QuantizeLinear/DequantizeLinear opset 23 scale dtype support. That issue is not about float8 values.
