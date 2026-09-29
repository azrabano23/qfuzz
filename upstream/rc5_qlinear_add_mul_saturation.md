# QDQ Add/Mul fused into QLinearAdd/QLinearMul saturates to the wrong end when the requantized value exceeds int32

> Paste-ready report for https://github.com/microsoft/onnxruntime/issues/new?template=01-bug_report.yml
> **Title:** `QLinearAdd/QLinearMul (and fused DQ->Add/Mul->Q) return 0 instead of 255 when (a op b)/c_scale > 2^31 (x86 MLAS)`
> Suggested labels: `core runtime`, `quantization`. Not filed yet.

### Describe the issue

The graph `DequantizeLinear(a) , DequantizeLinear(b) -> Add|Mul -> QuantizeLinear` uses standard ONNX operators only.
At `ORT_ENABLE_EXTENDED`/`ORT_ENABLE_ALL`, ORT's QDQ transformer fuses it into `com.microsoft.QLinearAdd` or
`QLinearMul` (verified by dumping the optimized model). When the exact result divided by the output scale exceeds
the int32 range, the fused kernel returns the type **minimum** (uint8: 0) instead of saturating to the maximum (255).
The unfused graph (`ORT_DISABLE_ALL`) returns 255, which is correct.

The expected value comes from composing the spec formulas (`onnx.defs.get_schema(...).doc`):
- DequantizeLinear: *"The dequantization formula is `y = (x - x_zero_point) * x_scale`."*
- QuantizeLinear: *"The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`. Saturation is done according to: [...] uint8: [0, 255]"*

With `a = 229, a_scale = 3000, b = 1, b_scale = 1, c_scale = 1e-4` (all zero points 0), Add gives
(687000 + 1) / 1e-4 = 6.87e9, and saturate() gives **255**. `ORT_DISABLE_ALL` returns 255, and the fused kernel returns **0**.
Apache TVM 0.27 (Relax ONNX frontend, `llvm` target) also returns `[0, 255, 255, 255]` for both models.

**Cause (from reading the source).** The x86 MLAS kernels convert the float result with `cvtps2dq` and only
saturate afterwards with `packs`/`packus`. Out-of-range lanes become `0x80000000` (INT32_MIN), which then saturates
to the minimum:
- QLinearAdd: [`onnxruntime/core/mlas/lib/intrinsics/avx2/qladd_avx2.cpp#L171-L172` @ v1.30.0](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/intrinsics/avx2/qladd_avx2.cpp#L171-L172)
  (`_mm256_packs_epi32(_mm256_cvtps_epi32(...))`), and the SSE2 kernel in
  [`qladd.cpp#L396-L400`](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/qladd.cpp#L396-L400);
- QLinearMul: [`qlmul.cpp#L213-L214`](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/qlmul.cpp#L213-L214)
  (`MlasQLinearMulVectorS16`).

The scalar fallbacks (`MlasQLinearAddKernelRawHelper`, the generic `MlasQLinearMulKernel`) clamp in float before
converting and are correct. Clamping the float vector to `[qmin - zp, qmax - zp]` (or to ±2^31) before `cvtps2dq`
would fix the vector paths.

**How likely is it in practice?** It needs `(input scale / output scale) * |value| > 2^31`. For uint8 that means a
scale ratio above about 8e6, as with a badly calibrated or degenerate output scale. In a 150k-case fuzzing campaign,
8 of 532 hits had all scales inside [1e-12, 1e12]. The rest also involved extreme scales. I still think it is worth
fixing, because the error is not small: the result saturates to the opposite end of the range.

**Note on `onnx.reference`.** `onnx.reference.ReferenceEvaluator` returns 0 here as well. Its QuantizeLinear does
`np.rint(x).astype(np.int32)` before clipping, which is the same class of bug (a separate report for onnx/onnx is
drafted next to this file). So the script uses the spec formula and `ORT_DISABLE_ALL` as the oracle and prints the
reference output for information only.

### To reproduce

The script needs only `onnx`, `onnxruntime` and `numpy`. It builds the model inline and exits with an
`AssertionError` while the bug is present.

```python
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
```

Output on onnxruntime 1.30.0 (PyPI wheel, `git-commit-id=f2c39fe`), on an Intel Xeon with AVX-512 VNNI/AMX:

```
onnxruntime 1.30.0 onnx 1.23.0
Add: exact (a*sa Add b*sb)/sc = [0.00000e+00 3.00200e+07 1.80001e+09 6.87001e+09]
Add: expected [  0 255 255 255]  onnx.reference [  0 255 255   0]  ORT_DISABLE_ALL [  0 255 255 255]  ORT_ENABLE_ALL [  0 255 255   0]
Mul: exact (a*sa Mul b*sb)/sc = [0.00e+00 6.00e+07 1.80e+09 6.87e+09]
Mul: expected [  0 255 255 255]  onnx.reference [  0 255 255   0]  ORT_DISABLE_ALL [  0 255 255 255]  ORT_ENABLE_ALL [  0 255 255   0]
Traceback (most recent call last):
  File "rc5_qlinear_add_mul_saturation.py", line 38, in <module>
    assert ok, "fused QLinearAdd/QLinearMul result differs from the ONNX spec (and from the unfused graph)"
           ^^
AssertionError: fused QLinearAdd/QLinearMul result differs from the ONNX spec (and from the unfused graph)
# (exit code 1)
```

`so.optimized_model_filepath` confirms that the optimized graph is a single `com.microsoft.QLinearAdd` (or
`QLinearMul`) node. Calling `com.microsoft.QLinearAdd` directly with the same scales gives the same wrong result.

### Urgency

Low. It needs extreme input/output scale ratios. The result is a silent sign-flip-like error, not a crash.

### Platform

Linux

### OS Version

Ubuntu 24.04.4 LTS (kernel 6.18), x86-64

### ONNX Runtime Installation

Released Package

### ONNX Runtime Version or Commit ID

1.30.0 (`git-commit-id=f2c39fe`). `qladd.cpp`, `qlmul.cpp` and `qlinear_binary_op.cc` are byte-identical on `main` as of
2026-09-29.

### ONNX Runtime API

Python

### Architecture

X64

### Execution Provider

Default CPU

### Execution Provider Library Version

_No response_

---

**Related issues and PRs (searched 2026-09-29 for "QLinearAdd QLinearMul saturation overflow"; none is a duplicate):**
- #32132 (open): fused QLinearAdd/QLinearConv/QLinearAveragePool off by 1-2 quantization steps near rounding ties.
  The operators are the same, but the effect is different: small rounding differences, not saturation to the wrong
  end.
- PR #32754 (open): fixes float32 overflow/underflow of the `ScaleA*ScaleB/ScaleC` ratio in the **ARM64 NEON**
  QLinearMul kernel. That is related (extreme scales), but it covers ARM only and the scale-ratio computation, not
  the x86 float to int32 conversion. Its description notes that the SSE2 kernel has the same ratio issue.
- #11759 (2022): QLinearAdd off by one. That is a rounding question, not a saturation one.
- #29727 (closed): QLinearSoftmax int8 output saturation. The class is similar, but the operator is different.
