# qfuzz campaign report

ONNX Runtime **1.30.0** (CPU EP), onnx 1.23.0, numpy 2.4.6, Python 3.11.15 on `Intel(R) Xeon(R) Processor @ 2.10GHz` (ISA: avx2, avx512f, avx512_vnni, avx_vnni, amx_int8, amx_tile).

Backend: **Apache TVM 0.27.0** (Relax ONNX frontend, target `llvm`).

**20000** generated cases, 2 workers, 2028.3 s wall clock, each executed at tvm and checked against the exact reference.

## Verdicts

| class | cases | share |
|---|---:|---:|
| within-bound | 1305 | 6.53% |
| exact | 15055 | 75.28% |
| tvm-error-build | 88 | 0.44% |
| tvm-error-import | 2219 | 11.10% |
| tvm-unsupported | 1333 | 6.67% |

TVM output bit-identical to ORT `ORT_DISABLE_ALL`: **16069** cases; differs: **257**; not compared (TVM or ORT error): 3674.

## Per pattern

| pattern | within-bound | exact | tvm-error-build | tvm-error-import | tvm-unsupported |
|---|---:|---:|---:|---:|---:|
| chain | 0 | 1659 | 7 | 0 | 0 |
| dequantize | 142 | 621 | 0 | 343 | 561 |
| dq_binary_q | 206 | 1461 | 0 | 0 | 0 |
| dq_conv_q | 69 | 1553 | 45 | 0 | 0 |
| dq_matmul_nbits | 0 | 0 | 0 | 1666 | 0 |
| dq_matmul_q | 89 | 1542 | 36 | 0 | 0 |
| dq_relu_q | 0 | 1667 | 0 | 0 | 0 |
| dyn_matmul | 21 | 1645 | 0 | 0 | 0 |
| dynamic_quantize | 126 | 1540 | 0 | 0 | 0 |
| matmul_integer | 0 | 1667 | 0 | 0 | 0 |
| qdq | 414 | 1253 | 0 | 0 | 0 |
| quantize | 238 | 447 | 0 | 210 | 772 |

## Raw signatures (16 distinct, before minimization)

| class | hits | signature | example seeds |
|---|---:|---|---|
| tvm-error-import | 1666 | `tvm-error-import\|DequantizeLinear+MatMul\|dq_matmul_nbits\|InternalError: Check failed: arr_size == nbytes (N vs` | 10, 22, 34 |
| tvm-unsupported | 368 | `tvm-unsupported\|DequantizeLinear\|dequantize\|TypeError: zero_point param datatype should be one of ['intN', 'ui` | 13, 25, 49 |
| tvm-error-import | 343 | `tvm-error-import\|DequantizeLinear\|dequantize\|InternalError: Check failed: arr_size == nbytes (N vs. N) : Tenso` | 73, 109, 133 |
| tvm-error-import | 210 | `tvm-error-import\|DequantizeLinear+QuantizeLinear\|quantize\|InternalError: Check failed: arr_size == nbytes (N v` | 12, 24, 192 |
| tvm-unsupported | 195 | `tvm-unsupported\|DequantizeLinear+QuantizeLinear\|quantize\|ValueError: QuantizeLinear floatN quantization with s` | 0, 96, 180 |
| tvm-unsupported | 194 | `tvm-unsupported\|DequantizeLinear+QuantizeLinear\|quantize\|TypeError: zero_point param datatype should be one of` | 228, 372, 468 |
| tvm-unsupported | 183 | `tvm-unsupported\|DequantizeLinear\|dequantize\|ValueError: DequantizeLinear blocked quantization is not supported` | 1, 181, 205 |
| tvm-unsupported | 161 | `tvm-unsupported\|QuantizeLinear\|quantize\|ValueError: QuantizeLinear blocked quantization is not supported yet.` | 204, 252, 420 |
| tvm-unsupported | 113 | `tvm-unsupported\|DequantizeLinear+QuantizeLinear\|quantize\|TypeError: Unsupported output datatype attribute for ` | 144, 432, 492 |
| tvm-unsupported | 107 | `tvm-unsupported\|DequantizeLinear+QuantizeLinear\|quantize\|TypeError: Unsupported output datatype attribute for ` | 156, 216, 276 |
| tvm-error-build | 36 | `tvm-error-build\|DequantizeLinear+MatMul+QuantizeLinear\|dq_matmul_q\|InternalError: Check failed: fb->value != N` | 1455, 2019, 3003 |
| tvm-error-build | 36 | `tvm-error-build\|Conv+DequantizeLinear+QuantizeLinear\|dq_conv_q\|InternalError: Check failed: fb->value != N (N ` | 1733, 2141, 2441 |
| tvm-unsupported | 10 | `tvm-unsupported\|DequantizeLinear\|dequantize\|TypeError: Unsupported input datatype for operation: floatN` | 3205, 9817, 12985 |
| tvm-error-build | 9 | `tvm-error-build\|Conv+DequantizeLinear+QuantizeLinear+Relu\|dq_conv_q\|InternalError: Check failed: fb->value != ` | 1385, 3113, 4253 |
| tvm-error-build | 7 | `tvm-error-build\|DequantizeLinear+MatMul+QuantizeLinear\|chain\|InternalError: Check failed: fb->value != N (N vs` | 1787, 14531, 14579 |
| tvm-unsupported | 2 | `tvm-unsupported\|DequantizeLinear+QuantizeLinear\|quantize\|ValueError: QuantizeLinear blocked quantization is no` | 3180, 6516 |

## TVM triage

Findings (mismatch / off-by-one / nan-inf / crash): **0**.

| group | verdict | cases |
|---|---|---:|
| T1: any int4/uint4 initializer fails to import (TensorCopyFromBytes size mismatch) | tvm-importer-bug | 2219 |
| T2: features the importer rejects explicitly (not bugs) | unsupported | 1333 |
| T3: y_scale = 0 fails at compile time (Divide by zero in constant folding) | spec-undefined | 88 |

### T1: any int4/uint4 initializer fails to import (TensorCopyFromBytes size mismatch)

`_parse_graph_initializers` turns an INT4/UINT4 TensorProto into an ml_dtypes int4 array via onnx.numpy_helper.to_array (1 byte per element) and wraps it in relax.const, whose TVM int4 tensor is packed (2 elements per byte). The byte counts differ, and the import aborts with `Check failed: arr_size == nbytes`. It happens for int32_data and raw_data encodings alike, so every blocked int4 weight graph (the DQ->MatMul 'MatMulNBits' pattern) is untestable. It is a loud failure on a valid model, and no wrong numbers come out. No matching apache/tvm issue was found (searched 2026-09-29); related: #20124 (ONNX op coverage RFC).

### T2: features the importer rejects explicitly (not bugs)

The importer raises a clear error for: blocked Q/DQ (`block_size>0`), float8 zero points in Q/DQ (E4M3FN/E5M2 are not in its allowed zero-point types), int4/uint4 as QuantizeLinear output, float8 `saturate=0`, and ops that have no converter (QLinearMatMul, QLinearConv, ConvInteger, com.microsoft.QLinearAdd/Mul). The TVM campaign avoids the last group by generating only patterns made of supported ops. The replay of ORT findings still hits it.

### T3: y_scale = 0 fails at compile time (Divide by zero in constant folding)

All 88 cases have a zero Q/DQ scale. The ONNX spec's formula divides by it, and qfuzz treats a zero scale as undefined (see README 'Spec ambiguities'). ORT accepts these models and returns some value (0 or -128 in the cases checked), and TVM refuses to compile them. Both are acceptable, so this is not a bug.

## ORT's 55 minimized findings replayed on TVM

| ORT root cause | TVM verdicts |
|---|---|
| RC1-double-qdq-remover | exact: 5 |
| RC2-matmulnbits-accuracy-level | tvm-error-import: 3 |
| RC3-float8-saturate-false | tvm-unsupported: 2 |
| RC4-blocked-q-int32-overflow | tvm-unsupported: 10 |
| RC5-qlinear-binary-overflow | exact: 7, tvm-unsupported: 1 |
| RC6-requant-float32-extreme-scales | exact: 8, tvm-unsupported: 11 |
| RC7-rejects-valid | exact: 1, tvm-unsupported: 7 |

