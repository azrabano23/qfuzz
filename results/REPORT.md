# qfuzz campaign report

ONNX Runtime **1.30.0** (CPU EP), onnx 1.23.0, numpy 2.4.6, Python 3.11.15 on `Intel(R) Xeon(R) Processor @ 2.10GHz` (ISA: avx2, avx512f, avx512_vnni, avx_vnni, amx_int8, amx_tile).

**150000** generated cases, 4 workers, 350.1 s wall clock, each executed at optimization levels disable, extended, all and checked against the exact reference.

## Verdicts

| class | cases | share |
|---|---:|---:|
| **nan-inf** | 519 | 0.35% |
| **mismatch** | 8279 | 5.52% |
| **off-by-one** | 296 | 0.20% |
| **rejects-valid-model** | 4492 | 2.99% |
| within-bound | 11889 | 7.93% |
| exact | 124525 | 83.02% |

Cases where fusion changed at least one output bit (oracle a, any level vs disable): **9727** (6.48%).

## Per pattern

| pattern | nan-inf | mismatch | off-by-one | rejects-valid-model | within-bound | exact |
|---|---:|---:|---:|---:|---:|---:|
| chain | 0 | 1 | 0 | 0 | 5 | 6082 |
| conv_integer | 0 | 0 | 0 | 1329 | 0 | 4768 |
| dequantize | 0 | 0 | 0 | 161 | 1143 | 11030 |
| dq_binary_q | 0 | 464 | 101 | 0 | 1564 | 6936 |
| dq_conv_q | 0 | 35 | 10 | 0 | 433 | 8738 |
| dq_matmul_nbits | 25 | 6163 | 0 | 0 | 9 | 2 |
| dq_matmul_q | 0 | 40 | 12 | 0 | 633 | 11592 |
| dq_relu_q | 0 | 0 | 0 | 0 | 0 | 6151 |
| dyn_matmul | 0 | 0 | 0 | 0 | 64 | 6086 |
| dynamic_quantize | 0 | 0 | 0 | 0 | 112 | 9068 |
| matmul_integer | 0 | 0 | 0 | 265 | 0 | 8895 |
| qdq | 1 | 652 | 0 | 0 | 2566 | 5925 |
| qlinear_binary | 0 | 130 | 81 | 0 | 7 | 8975 |
| qlinear_conv | 0 | 155 | 46 | 882 | 563 | 7466 |
| qlinear_matmul | 0 | 205 | 46 | 1568 | 224 | 10328 |
| quantize | 493 | 434 | 0 | 287 | 4566 | 12483 |

## Raw signatures (55 distinct, before minimization)

| class | hits | signature | example seeds |
|---|---:|---|---|
| nan-inf | 300 | `nan-inf\|all,disable,extended\|DequantizeLinear+QuantizeLinear\|float32\|normal\|quantize` | 7, 297, 392 |
| nan-inf | 193 | `nan-inf\|all,disable,extended\|DequantizeLinear+QuantizeLinear\|float32\|extreme-scale\|quantize` | 1335, 1865, 2536 |
| nan-inf | 25 | `nan-inf\|all,extended\|MatMul\|float32\|extreme-scale\|dq_matmul_nbits` | 2150, 3622, 5515 |
| nan-inf | 1 | `nan-inf\|all,extended\|DequantizeLinear+QuantizeLinear\|float32\|extreme-scale\|qdq` | 40813 |
| mismatch | 5198 | `mismatch\|all,extended\|MatMul\|float32\|normal\|dq_matmul_nbits` | 44, 53, 57 |
| mismatch | 965 | `mismatch\|all,extended\|MatMul\|float32\|extreme-scale\|dq_matmul_nbits` | 43, 454, 653 |
| mismatch | 435 | `mismatch\|all,extended\|DequantizeLinear+QuantizeLinear\|float32\|normal\|qdq` | 413, 640, 958 |
| mismatch | 140 | `mismatch\|all,disable,extended\|DequantizeLinear+QuantizeLinear\|float32\|extreme-scale\|quantize` | 1214, 3410, 3638 |
| mismatch | 133 | `mismatch\|all,extended\|Relu\|float32\|normal\|qdq` | 2131, 4647, 6910 |
| mismatch | 132 | `mismatch\|all,disable,extended\|QLinearMatMul\|uint8\|extreme-scale\|qlinear_matmul` | 544, 643, 2219 |
| mismatch | 128 | `mismatch\|all,extended\|Mul\|int8\|extreme-scale\|dq_binary_q` | 1161, 2534, 3413 |
| mismatch | 122 | `mismatch\|all,extended\|Mul\|uint8\|extreme-scale\|dq_binary_q` | 581, 1001, 1637 |
| mismatch | 114 | `mismatch\|all,disable,extended\|QLinearConv\|uint8\|extreme-scale\|qlinear_conv` | 1940, 1941, 3850 |
| mismatch | 113 | `mismatch\|all,extended\|Add\|int8\|extreme-scale\|dq_binary_q` | 96, 306, 1437 |
| mismatch | 93 | `mismatch\|all,extended\|Add\|uint8\|extreme-scale\|dq_binary_q` | 40, 214, 2294 |
| mismatch | 73 | `mismatch\|all,disable,extended\|QLinearMatMul\|int8\|extreme-scale\|qlinear_matmul` | 493, 734, 868 |
| mismatch | 68 | `mismatch\|all,disable,extended\|QLinearMul\|int8\|extreme-scale\|qlinear_binary` | 1164, 8196, 9035 |
| mismatch | 67 | `mismatch\|all,extended\|DequantizeLinear+QuantizeLinear\|float32\|extreme-scale\|qdq` | 1448, 3523, 6207 |
| mismatch | 64 | `mismatch\|all,disable,extended\|QuantizeLinear\|uint16\|extreme-scale\|quantize` | 2427, 3698, 4219 |
| mismatch | 62 | `mismatch\|all,disable,extended\|QLinearMul\|uint8\|extreme-scale\|qlinear_binary` | 5134, 5756, 5895 |
| mismatch | 60 | `mismatch\|all,disable,extended\|DequantizeLinear+QuantizeLinear\|float32\|normal\|quantize` | 2745, 3334, 5919 |
| mismatch | 53 | `mismatch\|all,disable,extended\|QuantizeLinear\|int16\|extreme-scale\|quantize` | 506, 990, 5236 |
| mismatch | 51 | `mismatch\|all,disable,extended\|QuantizeLinear\|uint8\|extreme-scale\|quantize` | 7939, 8173, 8318 |
| mismatch | 41 | `mismatch\|all,disable,extended\|QLinearConv\|int8\|extreme-scale\|qlinear_conv` | 9072, 14760, 15070 |
| mismatch | 40 | `mismatch\|all,extended\|MatMul\|uint8\|extreme-scale\|dq_matmul_q` | 564, 4533, 4883 |
| mismatch | 35 | `mismatch\|all,extended\|Conv\|uint8\|extreme-scale\|dq_conv_q` | 101, 5567, 6241 |
| mismatch | 32 | `mismatch\|all,disable,extended\|QuantizeLinear\|int8\|extreme-scale\|quantize` | 7628, 13032, 13076 |
| mismatch | 17 | `mismatch\|all,extended\|Relu\|float32\|extreme-scale\|qdq` | 12305, 17669, 19135 |
| mismatch | 12 | `mismatch\|all,disable,extended\|QuantizeLinear\|int8\|normal\|quantize` | 24342, 50648, 54578 |
| mismatch | 9 | `mismatch\|all,disable,extended\|QuantizeLinear\|uint16\|normal\|quantize` | 6799, 37758, 39378 |
| mismatch | 7 | `mismatch\|all,disable,extended\|QuantizeLinear\|int16\|normal\|quantize` | 1908, 26535, 37259 |
| mismatch | 6 | `mismatch\|all,disable,extended\|QuantizeLinear\|uint8\|normal\|quantize` | 8636, 28354, 43221 |
| mismatch | 4 | `mismatch\|all,extended\|Mul\|uint8\|normal\|dq_binary_q` | 12937, 15220, 45412 |
| mismatch | 3 | `mismatch\|all,extended\|Mul\|int8\|normal\|dq_binary_q` | 2735, 77611, 124775 |
| mismatch | 1 | `mismatch\|all,extended\|Add\|uint8\|normal\|dq_binary_q` | 2518 |
| mismatch | 1 | `mismatch\|all,extended\|MatMul\|uint8\|extreme-scale\|chain` | 58069 |
| off-by-one | 56 | `off-by-one\|all,extended\|Mul\|uint8\|extreme-scale\|dq_binary_q` | 1043, 16409, 19413 |
| off-by-one | 45 | `off-by-one\|all,extended\|Mul\|int8\|extreme-scale\|dq_binary_q` | 1071, 3227, 4302 |
| off-by-one | 42 | `off-by-one\|all,disable,extended\|QLinearMul\|uint8\|extreme-scale\|qlinear_binary` | 4373, 4884, 17957 |
| off-by-one | 39 | `off-by-one\|all,disable,extended\|QLinearMul\|int8\|extreme-scale\|qlinear_binary` | 4346, 4402, 9166 |
| off-by-one | 36 | `off-by-one\|all,disable,extended\|QLinearMatMul\|uint8\|extreme-scale\|qlinear_matmul` | 8223, 15925, 18168 |
| off-by-one | 31 | `off-by-one\|all,disable,extended\|QLinearConv\|uint8\|extreme-scale\|qlinear_conv` | 4211, 7556, 11177 |
| off-by-one | 15 | `off-by-one\|all,disable,extended\|QLinearConv\|int8\|extreme-scale\|qlinear_conv` | 7798, 13350, 17221 |
| off-by-one | 12 | `off-by-one\|all,extended\|MatMul\|uint8\|extreme-scale\|dq_matmul_q` | 19351, 19947, 23354 |
| off-by-one | 10 | `off-by-one\|all,disable,extended\|QLinearMatMul\|int8\|extreme-scale\|qlinear_matmul` | 450, 21285, 71728 |
| off-by-one | 9 | `off-by-one\|all,extended\|Conv\|uint8\|extreme-scale\|dq_conv_q` | 3780, 12299, 14443 |
| off-by-one | 1 | `off-by-one\|all,extended\|Relu\|uint8\|extreme-scale\|dq_conv_q` | 91996 |
| rejects-valid-model | 1329 | `rejects-valid-model\|all,disable,extended\|ConvInteger\|[ONNXRuntimeError] : N : FAIL : Non-zero status code retu` | 66, 139, 204 |
| rejects-valid-model | 1146 | `rejects-valid-model\|all,disable,extended\|QLinearMatMul\|[ONNXRuntimeError] : N : NOT_IMPLEMENTED : Could not fi` | 162, 341, 342 |
| rejects-valid-model | 882 | `rejects-valid-model\|all,disable,extended\|QLinearConv\|[ONNXRuntimeError] : N : NOT_IMPLEMENTED : Could not find` | 124, 254, 566 |
| rejects-valid-model | 422 | `rejects-valid-model\|all,disable,extended\|QLinearMatMul\|[ONNXRuntimeError] : N : FAIL : Non-zero status code re` | 144, 708, 719 |
| rejects-valid-model | 265 | `rejects-valid-model\|all,disable,extended\|MatMulInteger\|[ONNXRuntimeError] : N : FAIL : Non-zero status code re` | 446, 597, 622 |
| rejects-valid-model | 191 | `rejects-valid-model\|all,disable,extended\|QuantizeLinear\|[ONNXRuntimeError] : N : FAIL : Non-zero status code r` | 289, 588, 694 |
| rejects-valid-model | 161 | `rejects-valid-model\|all,disable,extended\|DequantizeLinear\|[ONNXRuntimeError] : N : FAIL : Non-zero status code` | 2691, 3059, 4199 |
| rejects-valid-model | 96 | `rejects-valid-model\|all,disable,extended\|DequantizeLinear+QuantizeLinear\|[ONNXRuntimeError] : N : FAIL : Non-z` | 235, 1113, 4618 |

## Minimized findings (55)

Deduplicated by the signature of the *minimized* graph. Triage verdicts are curated by hand in `results/triage.json` after reading the ONNX spec and the minimized repro.

### Root-cause summary

| root cause | verdict | minimized findings | raw hits (cases) | scale regimes |
|---|---|---:|---:|---|
| RC1: DoubleQDQPairsRemover re-quantizes Q1->DQ1->Q2->DQ2 onto a *new* grid (graph optimization  | Real ORT bug | 5 | 653 | extreme-scale, normal |
| RC2: DQ(int4 blocked)->MatMul is fused into MatMulNBits with int8-quantized activations by defa | Intentional ORT behaviour (documented trade-off), not spec-conformant | 3 | 6188 | extreme-scale, normal |
| RC3: QuantizeLinear to float8 with saturate=0 mis-handles the rounding band just above FLT_MAX | Real ORT bug | 2 | 493 | extreme-scale, normal |
| RC4: blocked (and 4-bit) QuantizeLinear saturates to the WRONG end when abs(x/scale) >= 2^31 (i | Real ORT bug | 10 | 434 | extreme-scale, normal |
| RC5: QLinearAdd/QLinearMul (incl. fused DQ->Add/Mul->Q) saturate to the wrong end when the requ | Real ORT bug | 8 | 532 | extreme-scale, normal |
| RC6: requantization multiplier folded in float32 breaks for extreme / subnormal scales (QLinear | Real ORT bug (only with degenerate/extreme inputs) | 19 | 794 | extreme-scale |
| RC7: valid models rejected (type combinations / per-row / per-channel zero points / blocked-wit | Valid model rejected (unsupported type/feature) | 8 | 4492 | - |

| finding | class | levels | graph | hits | triage | root cause |
|---|---|---|---|---:|---|---|
| [`mm-quantizelinear-dequantizelinear-quantize-ce328a`](../findings/mm-quantizelinear-dequantizelinear-quantize-ce328a/repro.py) | mismatch | all,extended | QuantizeLinear → DequantizeLinear → QuantizeLinear → DequantizeLinear | 435 | Real ORT bug | RC1-double-qdq-remover |
| [`mm-relu-f5f010`](../findings/mm-relu-f5f010/repro.py) | mismatch | all,extended | QuantizeLinear → DequantizeLinear → QuantizeLinear → DequantizeLinear → Relu | 133 | Real ORT bug | RC1-double-qdq-remover |
| [`mm-quantizelinear-dequantizelinear-quantize-ef0593`](../findings/mm-quantizelinear-dequantizelinear-quantize-ef0593/repro.py) | mismatch | all,extended | QuantizeLinear → DequantizeLinear → QuantizeLinear → DequantizeLinear | 67 | Real ORT bug | RC1-double-qdq-remover |
| [`mm-relu-f35585`](../findings/mm-relu-f35585/repro.py) | mismatch | all,extended | QuantizeLinear → DequantizeLinear → QuantizeLinear → DequantizeLinear → Relu | 17 | Real ORT bug | RC1-double-qdq-remover |
| [`nan-quantizelinear-dequantizelinear-quantize-fea0cb`](../findings/nan-quantizelinear-dequantizelinear-quantize-fea0cb/repro.py) | nan-inf | all,extended | QuantizeLinear → DequantizeLinear → QuantizeLinear → DequantizeLinear | 1 | Real ORT bug | RC1-double-qdq-remover |
| [`mm-matmul-f7b7d9`](../findings/mm-matmul-f7b7d9/repro.py) | mismatch | all,extended | DequantizeLinear → MatMul | 5198 | Intentional ORT behaviour (documented trade-off), not spec-conformant | RC2-matmulnbits-accuracy-level |
| [`mm-matmul-e6c919`](../findings/mm-matmul-e6c919/repro.py) | mismatch | all,extended | DequantizeLinear → MatMul | 965 | Intentional ORT behaviour (documented trade-off), not spec-conformant | RC2-matmulnbits-accuracy-level |
| [`nan-matmul-347847`](../findings/nan-matmul-347847/repro.py) | nan-inf | all,extended | DequantizeLinear → MatMul | 25 | Intentional ORT behaviour (documented trade-off), not spec-conformant | RC2-matmulnbits-accuracy-level |
| [`nan-quantizelinear-dequantizelinear-4686bf`](../findings/nan-quantizelinear-dequantizelinear-4686bf/repro.py) | nan-inf | all,disable,extended | QuantizeLinear → DequantizeLinear | 300 | Real ORT bug | RC3-float8-saturate-false |
| [`nan-quantizelinear-dequantizelinear-55b8e3`](../findings/nan-quantizelinear-dequantizelinear-55b8e3/repro.py) | nan-inf | all,disable,extended | QuantizeLinear → DequantizeLinear | 193 | Real ORT bug | RC3-float8-saturate-false |
| [`mm-quantizelinear-dequantizelinear-6963b7`](../findings/mm-quantizelinear-dequantizelinear-6963b7/repro.py) | mismatch | all,disable,extended | QuantizeLinear → DequantizeLinear | 140 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-f7b541`](../findings/mm-quantizelinear-f7b541/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 64 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-dequantizelinear-c2024e`](../findings/mm-quantizelinear-dequantizelinear-c2024e/repro.py) | mismatch | all,disable,extended | QuantizeLinear → DequantizeLinear | 60 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-1c3dbc`](../findings/mm-quantizelinear-1c3dbc/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 53 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-162487`](../findings/mm-quantizelinear-162487/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 51 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-cb55fa`](../findings/mm-quantizelinear-cb55fa/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 32 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-b7b255`](../findings/mm-quantizelinear-b7b255/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 12 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-36bc7e`](../findings/mm-quantizelinear-36bc7e/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 9 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-0e4b8c`](../findings/mm-quantizelinear-0e4b8c/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 7 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-quantizelinear-744260`](../findings/mm-quantizelinear-744260/repro.py) | mismatch | all,disable,extended | QuantizeLinear | 6 | Real ORT bug | RC4-blocked-q-int32-overflow |
| [`mm-mul-6eee63`](../findings/mm-mul-6eee63/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 128 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-mul-302968`](../findings/mm-mul-302968/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 122 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-add-d14801`](../findings/mm-add-d14801/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Add → QuantizeLinear | 113 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-add-0d23de`](../findings/mm-add-0d23de/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Add → QuantizeLinear | 93 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-qlinearmul-408803`](../findings/mm-qlinearmul-408803/repro.py) | mismatch | all,disable,extended | QLinearMul | 68 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-mul-0a9a3d`](../findings/mm-mul-0a9a3d/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 4 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-mul-670d5f`](../findings/mm-mul-670d5f/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 3 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-add-c06392`](../findings/mm-add-c06392/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → Add → QuantizeLinear | 1 | Real ORT bug | RC5-qlinear-binary-overflow |
| [`mm-qlinearmatmul-2baa28`](../findings/mm-qlinearmatmul-2baa28/repro.py) | mismatch | all,disable,extended | QLinearMatMul | 132 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-qlinearconv-32c8d6`](../findings/mm-qlinearconv-32c8d6/repro.py) | mismatch | all,disable,extended | QLinearConv | 114 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-qlinearmatmul-19218d`](../findings/mm-qlinearmatmul-19218d/repro.py) | mismatch | all,disable,extended | QLinearMatMul | 73 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-qlinearmul-c80736`](../findings/mm-qlinearmul-c80736/repro.py) | mismatch | all,disable,extended | QLinearMul | 62 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-mul-0602d8`](../findings/obo-mul-0602d8/repro.py) | off-by-one | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 56 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-mul-61f545`](../findings/obo-mul-61f545/repro.py) | off-by-one | all,extended | DequantizeLinear → DequantizeLinear → Mul → QuantizeLinear | 45 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearmul-634349`](../findings/obo-qlinearmul-634349/repro.py) | off-by-one | all,disable,extended | QLinearMul | 42 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-qlinearconv-dab856`](../findings/mm-qlinearconv-dab856/repro.py) | mismatch | all,disable,extended | QLinearConv | 41 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-matmul-ef058a`](../findings/mm-matmul-ef058a/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → MatMul → QuantizeLinear | 40 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearmul-b4fab1`](../findings/obo-qlinearmul-b4fab1/repro.py) | off-by-one | all,disable,extended | QLinearMul | 39 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearmatmul-31f3be`](../findings/obo-qlinearmatmul-31f3be/repro.py) | off-by-one | all,disable,extended | QLinearMatMul | 36 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-conv-5acfa0`](../findings/mm-conv-5acfa0/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → DequantizeLinear → Conv → QuantizeLinear | 35 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearconv-6c81e2`](../findings/obo-qlinearconv-6c81e2/repro.py) | off-by-one | all,disable,extended | QLinearConv | 31 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearconv-a5e2a9`](../findings/obo-qlinearconv-a5e2a9/repro.py) | off-by-one | all,disable,extended | QLinearConv | 15 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-matmul-7b6d61`](../findings/obo-matmul-7b6d61/repro.py) | off-by-one | all,extended | DequantizeLinear → DequantizeLinear → MatMul → QuantizeLinear | 12 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-qlinearmatmul-108f5b`](../findings/obo-qlinearmatmul-108f5b/repro.py) | off-by-one | all,disable,extended | QLinearMatMul | 10 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-conv-8de96b`](../findings/obo-conv-8de96b/repro.py) | off-by-one | all,extended | DequantizeLinear → DequantizeLinear → Conv → QuantizeLinear | 9 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`mm-matmul-6a9f21`](../findings/mm-matmul-6a9f21/repro.py) | mismatch | all,extended | DequantizeLinear → DequantizeLinear → MatMul → QuantizeLinear | 1 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`obo-conv-relu-7a6628`](../findings/obo-conv-relu-7a6628/repro.py) | off-by-one | all,extended | DequantizeLinear → DequantizeLinear → DequantizeLinear → Conv → Relu → QuantizeLinear | 1 | Real ORT bug (only with degenerate/extreme inputs) | RC6-requant-float32-extreme-scales |
| [`rej-convinteger-f1891a`](../findings/rej-convinteger-f1891a/repro.py) | rejects-valid-model | all,disable,extended | ConvInteger | 1329 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-qlinearmatmul-834f6d`](../findings/rej-qlinearmatmul-834f6d/repro.py) | rejects-valid-model | all,disable,extended | QLinearMatMul | 1146 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-qlinearconv-e7eef0`](../findings/rej-qlinearconv-e7eef0/repro.py) | rejects-valid-model | all,disable,extended | QLinearConv | 882 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-qlinearmatmul-65200e`](../findings/rej-qlinearmatmul-65200e/repro.py) | rejects-valid-model | all,disable,extended | QLinearMatMul | 422 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-matmulinteger-76e216`](../findings/rej-matmulinteger-76e216/repro.py) | rejects-valid-model | all,disable,extended | MatMulInteger | 265 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-quantizelinear-286034`](../findings/rej-quantizelinear-286034/repro.py) | rejects-valid-model | all,disable,extended | QuantizeLinear | 191 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-dequantizelinear-872574`](../findings/rej-dequantizelinear-872574/repro.py) | rejects-valid-model | all,disable,extended | DequantizeLinear | 161 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |
| [`rej-quantizelinear-dequantizelinear-436772`](../findings/rej-quantizelinear-dequantizelinear-436772/repro.py) | rejects-valid-model | all,disable,extended | QuantizeLinear → DequantizeLinear | 96 | Valid model rejected (unsupported type/feature) | RC7-rejects-valid |

### RC1: DoubleQDQPairsRemover re-quantizes Q1->DQ1->Q2->DQ2 onto a *new* grid (graph optimization changes results)

**Verdict:** Real ORT bug.  
**Known upstream?** No matching issue found (searched 'DoubleQDQPairsRemover', 'double QDQ'); related but different: [#32132](https://github.com/microsoft/onnxruntime/issues/32132) (fused QLinear* off by k steps), [#28030](https://github.com/microsoft/onnxruntime/issues/28030) (double QDQ crash, closed).  

At ORT_ENABLE_EXTENDED/ALL the `DoubleQDQPairsRemover` replaces Q1(s1,z1)->DQ1->Q2(s2,z2)->DQ2 by a single Q->DQ pair whose scale is `(min(real_max1,real_max2)-max(real_min1,real_min2))/(qmax-qmin)` (onnxruntime/core/optimizer/double_qdq_pairs_remover.cc, FindNewZeroPointAndScale). That new grid is neither s1 nor s2, so values that the original graph snaps to the s1 grid (or to the coarser s2 grid) come out different: e.g. int16, x=-6.504, s1=1, s2=0.001 gives -7.0 unoptimized (spec) but -6.504 optimized; uint16 x=20000, s1=s2=1, z2=30458 gives 20000 vs 19999.8. When the two pairs have equal ranges but different zero points the result is off by up to several LSBs. The ONNX spec defines the graph as the composition of the four ops, so the optimized result is non-conforming. Realistic scales; extreme-scale variants additionally overflow the new-scale computation (inf/NaN outputs for x=0).

- `mm-quantizelinear-dequantizelinear-quantize-ce328a` (435 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0, 0] got **-6.504000186920166**, exact **-7.000000476837158**, allowed slack 8.344650664904307e-07
  ```
  input x1: float32[1, 1, 1, 1] = [-6.50436926]
  const c2: float32[] = [1]
  const c3: int16[] = [0]
  const c6: float32[] = [0.00100000005]
  const c7: int16[] = [0]
  t4 = QuantizeLinear(x1, c2, c3)
  t5 = DequantizeLinear(t4, c2, c3)
  t8 = QuantizeLinear(t5, c6, c7)
  t9 = DequantizeLinear(t8, c6, c7)
  outputs: t9
  ```
- `mm-relu-f5f010` (133 raw hits; fused at extended: `QuantizeLinear DequantizeLinear Relu`)
  - example: index [0, 0, 0, 0] got **19999.8046875**, exact **20000**, allowed slack 0.009645938873291016
  ```
  input x1: float32[1, 1, 1, 1] = [20000]
  const c2: float32[] = [1]
  const c3: uint16[] = [0]
  const c6: float32[] = [1]
  const c7: uint16[] = [30458]
  t4 = QuantizeLinear(x1, c2, c3)
  t5 = DequantizeLinear(t4, c2, c3)
  t8 = QuantizeLinear(t5, c6, c7)
  t9 = DequantizeLinear(t8, c6, c7)
  t10 = Relu(t9)
  outputs: t10
  ```
- `mm-quantizelinear-dequantizelinear-quantize-ef0593` (67 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0, 0] got **-8.960000118690815e-30**, exact **0**, allowed slack 1.401298464324817e-45
  ```
  input x1: float32[1, 1, 1, 1] = [-1.70000005]
  const c2: float32[] = [4]
  const c3: int8[] = [0]
  const c6: float32[] = [7.00000009e-32]
  const c7: int8[] = [0]
  t4 = QuantizeLinear(x1, c2, c3)
  t5 = DequantizeLinear(t4, c2, c3)
  t8 = QuantizeLinear(t5, c6, c7)
  t9 = DequantizeLinear(t8, c6, c7)
  outputs: t9
  ```
- `mm-relu-f35585` (17 raw hits; fused at extended: `QuantizeLinear DequantizeLinear Relu`)
  - example: index [0, 0, 0, 0] got **9.99938558172903e-41**, exact **0**, allowed slack 1.401298464324817e-45
  ```
  input x1: float32[1, 1, 1, 1] = [9.9999461e-41]
  const c2: float32[] = [1.96181785e-44]
  const c3: int16[] = [0]
  const c6: float32[] = [1]
  const c7: int16[] = [0]
  t4 = QuantizeLinear(x1, c2, c3)
  t5 = DequantizeLinear(t4, c2, c3)
  t8 = QuantizeLinear(t5, c6, c7)
  t9 = DequantizeLinear(t8, c6, c7)
  t10 = Relu(t9)
  outputs: t10
  ```
- `nan-quantizelinear-dequantizelinear-quantize-fea0cb` (1 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0, 0] got **nan**, exact **0**, allowed slack 1.401298464324817e-45
  ```
  input x1: float32[1, 1, 1, 1] = [0]
  const c2: float32[] = [3.00000012e+34]
  const c3: uint16[] = [0]
  const c6: float32[] = [1.99999996e+34]
  const c7: uint16[] = [0]
  t4 = QuantizeLinear(x1, c2, c3)
  t5 = DequantizeLinear(t4, c2, c3)
  t8 = QuantizeLinear(t5, c6, c7)
  t9 = DequantizeLinear(t8, c6, c7)
  outputs: t9
  ```

### RC2: DQ(int4 blocked)->MatMul is fused into MatMulNBits with int8-quantized activations by default

**Verdict:** Intentional ORT behaviour (documented trade-off), not spec-conformant.  
**Known upstream?** Behaviour is controlled by session option `session.qdq_matmulnbits_accuracy_level` (default 4). Accuracy impact on LLMs reported in [#29849](https://github.com/microsoft/onnxruntime/issues/29849) (open).  

At ORT_ENABLE_EXTENDED/ALL, DequantizeLinear(int4/uint4, blocked, axis=0) feeding a float MatMul is replaced by com.microsoft.MatMulNBits with accuracy_level=4, which quantizes the *float activations* to int8 per block. The unoptimized graph (and the spec) compute a float32 MatMul, so the optimized result deviates by far more than float32 rounding (87 sampled realistic-scale cases: largest error per case relative to the largest output magnitude has a median of 0.6% and a maximum of 6.5%). Setting `session.qdq_matmulnbits_accuracy_level` to 0 removes every deviation (verified). It is a documented speed/accuracy trade-off, but it is applied silently by a graph optimization level that is on by default - worth a clear note, not a kernel bug.

- `mm-matmul-f7b7d9` (5198 raw hits; fused at extended: `com.microsoft.MatMulNBits`)
  - example: index [0, 0] got **-2.3937008380889893**, exact **-2.4000000953674316**, allowed slack 9.870529204647482e-06
  ```
  input a5: float32[1, 64] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const w1: uint4[64, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[2, 1] = [1, 1]
  const c3: uint4[2, 1] = [0, 4]
  t4 = DequantizeLinear(w1, c2, c3) {'axis': 0, 'block_size': 32}
  t6 = MatMul(a5, t4)
  outputs: t6
  ```
- `mm-matmul-e6c919` (965 raw hits; fused at extended: `com.microsoft.MatMulNBits`)
  - example: index [0, 4] got **0.8929133415222168**, exact **0.9000000357627869**, allowed slack 4.18961064774237e-05
  ```
  input a5: float32[1, 256] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const w1: uint4[256, 5] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[4, 5] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.00000002e-32 ...]
  const c3: uint4[4, 5] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  t4 = DequantizeLinear(w1, c2, c3) {'axis': 0, 'block_size': 64}
  t6 = MatMul(a5, t4)
  outputs: t6
  ```
- `nan-matmul-347847` (25 raw hits; fused at extended: `com.microsoft.MatMulNBits`)
  - example: index [0, 9] got **-inf**, exact **nan**, allowed slack 0
  ```
  input a5: float32[1, 64] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const w1: uint4[64, 32] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[4, 32] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ...]
  const c3: uint4[4, 32] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  t4 = DequantizeLinear(w1, c2, c3) {'axis': 0, 'block_size': 16}
  t6 = MatMul(a5, t4)
  outputs: t6
  ```

### RC3: QuantizeLinear to float8 with saturate=0 mis-handles the rounding band just above FLT_MAX

**Verdict:** Real ORT bug.  
**Known upstream?** No matching issue found (searched 'float8 saturate false', 'E5M2 NaN instead of infinity'); [#16938](https://github.com/microsoft/onnxruntime/issues/16938) is a different float8 off-by-one.  

ONNX Cast/QuantizeLinear float8 table (saturate=False): `[x] > FLT_MAX -> NaN` for E4M3FN and `-> Inf` for E5M2, where [x] is x rounded (RNE) to the target mantissa. onnx.reference agrees. ORT 1.30 CPU: E4M3FN maps |x/scale| in [480, 496) to +/-448 (a finite value) although 479 and 496 map to NaN - non-monotonic and contradicting the table; E5M2 maps (61440, 65536) to NaN instead of +/-Inf. Swept exhaustively in 0.25/16 steps; same for per-tensor, per-axis and blocked.

- `nan-quantizelinear-dequantizelinear-4686bf` (300 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0, 0] got **-448**, exact **nan**, allowed slack 0
  ```
  input x1: float32[1, 1, 1, 1] = [-0.0399999991]
  const c2: float32[] = [8.19999987e-05]
  const c3: float8e4m3fn[] = [0]
  const obs_s5: float32[] = [1]
  const obs_z6: float8e4m3fn[] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'saturate': 0}
  t7 = DequantizeLinear(t4, obs_s5, obs_z6)
  outputs: t7
  ```
- `nan-quantizelinear-dequantizelinear-55b8e3` (193 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0] got **448**, exact **nan**, allowed slack 0
  ```
  input x1: float32[1, 1, 1] = [1.80000005e+22]
  const c2: float32[] = [3.7e+19]
  const c3: float8e4m3fn[] = [0]
  const obs_s5: float32[] = [1]
  const obs_z6: float8e4m3fn[] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'saturate': 0}
  t7 = DequantizeLinear(t4, obs_s5, obs_z6)
  outputs: t7
  ```

### RC4: blocked (and 4-bit) QuantizeLinear saturates to the WRONG end when abs(x/scale) >= 2^31 (incl. x=+inf)

**Verdict:** Real ORT bug.  
**Known upstream?** No matching issue found (searched 'QuantizeLinear block_size saturation', 'int4 wrong saturation').  

Spec: y = saturate(round(x/scale)+zp). ORT returns the type *minimum* (or, with a negative zero point, the maximum) when |x/scale| >= 2^31 or x=+/-inf, e.g. uint8, blocked along a non-innermost axis, x=+inf -> 0 instead of 255; int4 blocked, x=-3e38, zp=-7 -> +7 instead of -8. Layout matters (probed with x=+inf, scale 1, zp 0): uint8/int8/int16/uint16 are wrong when the blocked axis is not the innermost one and right otherwise; int4/uint4 are also wrong for per-axis quantization of the innermost axis. Per-tensor is always right. Values below 2^31 are always right, which points at a float->int32 conversion before clamping (x86 cvtps2dq returns 0x80000000 for out-of-range input) in the strided/scalar code path - a hypothesis, not verified in the source. +inf activations (fp overflow upstream) make this reachable with ordinary scales.

- `mm-quantizelinear-dequantizelinear-6963b7` (140 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0] got **-8**, exact **7**, allowed slack 8.344650268554688e-07
  ```
  input x1: float32[1, 1, 1] = [inf]
  const c2: float32[1, 1, 1] = [7.99999995e+37]
  const c3: int4[1, 1, 1] = [0]
  const obs_s5: float32[] = [1]
  const obs_z6: int4[] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 2}
  t7 = DequantizeLinear(t4, obs_s5, obs_z6)
  outputs: t7
  ```
- `mm-quantizelinear-f7b541` (64 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 1, 1, 0] got **0**, exact **65535**, allowed slack 0
  ```
  input x1: float32[1, 2, 2, 1] = [0, 0, 0, inf]
  const c2: float32[1, 1, 2, 1] = [1, 6.00000021e+37]
  const c3: uint16[1, 1, 2, 1] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 1, 'block_size': 3}
  outputs: t4
  ```
- `mm-quantizelinear-dequantizelinear-c2024e` (60 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - example: index [0, 0, 0, 0] got **7**, exact **-8**, allowed slack 9.5367431640625e-07
  ```
  input x1: float32[1, 1, 1, 1] = [-3.00000001e+38]
  const c2: float32[1, 1, 1, 1] = [1]
  const c3: int4[1, 1, 1, 1] = [-7]
  const obs_s5: float32[] = [1]
  const obs_z6: int4[] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 2, 'block_size': 3}
  t7 = DequantizeLinear(t4, obs_s5, obs_z6)
  outputs: t7
  ```
- `mm-quantizelinear-1c3dbc` (53 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 0, 1, 0] got **-32768**, exact **32767**, allowed slack 0
  ```
  input x1: float32[1, 1, 2, 1] = [0, inf]
  const c2: float32[1, 1, 2, 1] = [1, 1.99999999e+37]
  const c3: int16[1, 1, 2, 1] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 4}
  outputs: t4
  ```
- `mm-quantizelinear-162487` (51 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 6, 1, 0] got **0**, exact **255**, allowed slack 0
  ```
  input x1: float32[1, 7, 2, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[1, 1, 2, 1] = [1, 3.00000004e+36]
  const c3: uint8[1, 1, 2, 1] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 1, 'block_size': 16}
  outputs: t4
  ```
- `mm-quantizelinear-cb55fa` (32 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 0, 2, 0] got **-128**, exact **127**, allowed slack 0
  ```
  input x1: float32[1, 1, 3, 2] = [0, 0, 0, 0, inf, 0]
  const c2: float32[1, 1, 1, 2] = [1.99999999e+37, 1]
  const c3: int8[1, 1, 1, 2] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 2, 'block_size': 16}
  outputs: t4
  ```
- `mm-quantizelinear-b7b255` (12 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [4, 0, 0] got **-128**, exact **127**, allowed slack 0
  ```
  input x1: float32[8, 2, 1] = [0, 0, 0, 0, 0, 0, 0, 0, inf, 0, 0, 0 ...]
  const c2: float32[1, 2, 1] = [1, 1]
  const c3: int8[1, 2, 1] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 8}
  outputs: t4
  ```
- `mm-quantizelinear-36bc7e` (9 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 0, 1] got **0**, exact **65535**, allowed slack 0
  ```
  input x1: float32[1, 1, 2] = [0, 3.00000001e+38]
  const c2: float32[1, 1, 2] = [1, 1]
  const c3: uint16[1, 1, 2] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 1, 'block_size': 1}
  outputs: t4
  ```
- `mm-quantizelinear-0e4b8c` (7 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 0, 0] got **-32768**, exact **32767**, allowed slack 0
  ```
  input x1: float32[1, 2, 2] = [3.00000001e+38, 0, 0, 0]
  const c2: float32[1, 1, 2] = [1, 1]
  const c3: int16[1, 1, 2] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 1, 'block_size': 2}
  outputs: t4
  ```
- `mm-quantizelinear-744260` (6 raw hits; fused at extended: `QuantizeLinear`)
  - example: index [0, 0, 0, 0] got **0**, exact **255**, allowed slack 0
  ```
  input x1: float32[1, 2, 2, 1] = [3.00000001e+38, 0, 0, 0]
  const c2: float32[1, 1, 2, 1] = [1, 1]
  const c3: uint8[1, 1, 2, 1] = [0, 0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 1, 'block_size': 2}
  outputs: t4
  ```

### RC5: QLinearAdd/QLinearMul (incl. fused DQ->Add/Mul->Q) saturate to the wrong end when the requantized value exceeds int32

**Verdict:** Real ORT bug.  
**Known upstream?** No matching issue found (searched 'QLinearAdd QLinearMul saturation overflow').  

com.microsoft.QLinearAdd/QLinearMul (used by ORT's QDQ fusion of DQ->Add/Mul->Q at EXTENDED/ALL) return qmin instead of qmax (or vice versa) when (A*B or A+B)/C_scale is beyond ~2^31, e.g. uint8 a=229, a_scale=3000, c_scale=1e-4: exact 6.9e9 saturates to 255, ORT returns 0. The unfused graph is correct, so the fusion changes the result from 255 to 0. Requires a large ratio between input and output scales (badly calibrated model) - plausible but uncommon: only 8 of the 532 campaign hits used scales inside [1e-12, 1e12]; the rest overlap with the degenerate-scale regime of RC6.

- `mm-mul-6eee63` (128 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0] got **-128**, exact **127**, allowed slack 0
  ```
  input a1: int8[1] = [107]
  input b5: int8[1] = [0]
  const c2: float32[] = [1]
  const c3: int8[] = [0]
  const c6: float32[] = [2.99999998e+25]
  const c7: int8[] = [-128]
  const c10: float32[] = [1]
  const c11: int8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-mul-302968` (122 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0] got **0**, exact **255**, allowed slack 0
  ```
  input a1: uint8[1] = [153]
  input b5: uint8[1] = [150]
  const c2: float32[] = [4.00000008e+20]
  const c3: uint8[] = [0]
  const c6: float32[] = [1]
  const c7: uint8[] = [0]
  const c10: float32[] = [1]
  const c11: uint8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-add-d14801` (113 raw hits; fused at extended: `com.microsoft.QLinearAdd`)
  - example: index [0] got **-128**, exact **127**, allowed slack 0
  ```
  input a1: int8[1] = [110]
  input b5: int8[1] = [0]
  const c2: float32[] = [8.00000033e+35]
  const c3: int8[] = [0]
  const c6: float32[] = [1]
  const c7: int8[] = [0]
  const c10: float32[] = [1]
  const c11: int8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Add(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-add-0d23de` (93 raw hits; fused at extended: `com.microsoft.QLinearAdd`)
  - example: index [0] got **0**, exact **255**, allowed slack 0
  ```
  input a1: uint8[1] = [198]
  input b5: uint8[1] = [0]
  const c2: float32[] = [1]
  const c3: uint8[] = [0]
  const c6: float32[] = [1]
  const c7: uint8[] = [0]
  const c10: float32[] = [2.99999784e-40]
  const c11: uint8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Add(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-qlinearmul-408803` (68 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0, 0] got **-128**, exact **127**, allowed slack 0
  ```
  input a1: int8[1, 1, 1] = [0]
  input b4: int8[1, 1, 1] = [127]
  const c2: float32[] = [1]
  const c3: int8[] = [-127]
  const c5: float32[] = [4.99999978e+24]
  const c6: int8[] = [0]
  const c7: float32[] = [1]
  const c8: int8[] = [0]
  t9 = com.microsoft.QLinearMul(a1, c2, c3, b4, c5, c6, c7, c8)
  outputs: t9
  ```
- `mm-mul-0a9a3d` (4 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0] got **0**, exact **255**, allowed slack 0
  ```
  input a1: uint8[1, 1] = [0]
  input b5: uint8[1, 1] = [0]
  const c2: float32[] = [1]
  const c3: uint8[] = [81]
  const c6: float32[] = [3000]
  const c7: uint8[] = [251]
  const c10: float32[] = [5.99999985e-05]
  const c11: uint8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-mul-670d5f` (3 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0, 0] got **-128**, exact **127**, allowed slack 0
  ```
  input a1: int8[1, 1, 1] = [0]
  input b5: int8[] = [0]
  const c2: float32[] = [5000]
  const c3: int8[] = [68]
  const c6: float32[] = [200]
  const c7: int8[] = [94]
  const c10: float32[] = [1]
  const c11: int8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-add-c06392` (1 raw hits; fused at extended: `com.microsoft.QLinearAdd`)
  - example: index [0, 0, 0] got **0**, exact **255**, allowed slack 0
  ```
  input a1: uint8[1, 1, 1] = [229]
  input b5: uint8[1] = [0]
  const c2: float32[] = [3000]
  const c3: uint8[] = [0]
  const c6: float32[] = [1]
  const c7: uint8[] = [0]
  const c10: float32[] = [9.99999975e-05]
  const c11: uint8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Add(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```

### RC6: requantization multiplier folded in float32 breaks for extreme / subnormal scales (QLinearMatMul, QLinearConv, QLinearMul, fused variants)

**Verdict:** Real ORT bug (only with degenerate/extreme inputs).  
**Known upstream?** Not searched further - degenerate inputs; related saturation class: [#29727](https://github.com/microsoft/onnxruntime/issues/29727) (QLinearSoftmax saturation, closed).  

ORT folds a_scale*b_scale/y_scale into one float32 multiplier. With scales outside ~[1e-12,1e12] the product under/overflows (subnormal -> few mantissa bits, or inf -> 0*inf = NaN -> garbage int), producing off-by-one/two results far from any rounding tie, or even qmin for acc=0 (expected: the zero point). The spec defines the exact result, so these are real non-conformances, but only with scales no real quantizer emits. Reported for completeness; low priority.

- `mm-qlinearmatmul-2baa28` (132 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0, 0] got **168**, exact **166**, allowed slack 0
  ```
  input a1: uint8[1, 1, 1] = [108]
  const c2: float32[] = [0.00079999998]
  const c3: uint8[] = [128]
  const w4: uint8[1, 1] = [255]
  const c5: float32[1] = [1.99993317e-41]
  const c6: uint8[1] = [242]
  const c7: float32[] = [6.0255834e-44]
  const c8: uint8[] = [235]
  t9 = QLinearMatMul(a1, c2, c3, w4, c5, c6, c7, c8)
  outputs: t9
  ```
- `mm-qlinearconv-32c8d6` (114 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 0, 0, 0] got **237**, exact **234**, allowed slack 0
  ```
  input x1: uint8[1, 2, 1, 2] = [255, 0, 57, 0]
  const c2: float32[] = [4.7499814e-41]
  const c3: uint8[] = [128]
  const w4: int8[1, 2, 1, 2] = [-112, -127, -51, 56]
  const c5: float32[1] = [0.000899999985]
  const c6: int8[1] = [0]
  const c7: float32[] = [7.00649232e-44]
  const c8: uint8[] = [0]
  const c9: int32[1] = [1898]
  t10 = QLinearConv(x1, c2, c3, w4, c5, c6, c7, c8, c9) {'kernel_shape': [1, 2]}
  outputs: t10
  ```
- `mm-qlinearmatmul-19218d` (73 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **0**, exact **127**, allowed slack 0
  ```
  input a1: int8[1, 1] = [0]
  const c2: float32[] = [0.00899999961]
  const c3: int8[] = [104]
  const w4: int8[1, 1] = [0]
  const c5: float32[1] = [2.94272678e-44]
  const c6: int8[1] = [81]
  const c7: float32[] = [7.00649232e-45]
  const c8: int8[] = [0]
  t9 = QLinearMatMul(a1, c2, c3, w4, c5, c6, c7, c8)
  outputs: t9
  ```
- `mm-qlinearmul-c80736` (62 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0] got **11**, exact **9**, allowed slack 0
  ```
  input a1: uint8[1, 1] = [63]
  input b4: uint8[1, 1] = [169]
  const c2: float32[] = [0.0046000001]
  const c3: uint8[] = [208]
  const c5: float32[] = [1.19993188e-41]
  const c6: uint8[] = [0]
  const c7: float32[] = [1.1399563e-41]
  const c8: uint8[] = [128]
  t9 = com.microsoft.QLinearMul(a1, c2, c3, b4, c5, c6, c7, c8)
  outputs: t9
  ```
- `obo-mul-0602d8` (56 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0] got **64**, exact **63**, allowed slack 0
  ```
  input a1: uint8[1] = [184]
  input b5: uint8[1] = [118]
  const c2: float32[] = [3.86310863]
  const c3: uint8[] = [160]
  const c6: float32[] = [2.0949412e-42]
  const c7: uint8[] = [226]
  const c10: float32[] = [5.17919912e-40]
  const c11: uint8[] = [104]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `obo-mul-61f545` (45 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0] got **92**, exact **93**, allowed slack 0
  ```
  input a1: int8[1, 1] = [61]
  input b5: int8[1, 1] = [-30]
  const c2: float32[] = [6.68000379e-40]
  const c3: int8[] = [0]
  const c6: float32[] = [0.000243999995]
  const c7: int8[] = [-127]
  const c10: float32[] = [1.04004372e-41]
  const c11: int8[] = [0]
  t4 = DequantizeLinear(a1, c2, c3)
  t8 = DequantizeLinear(b5, c6, c7)
  t9 = Mul(t4, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `obo-qlinearmul-634349` (42 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0] got **125**, exact **124**, allowed slack 0
  ```
  input a1: uint8[1] = [255]
  input b4: uint8[1] = [245]
  const c2: float32[] = [9.99966584e-42]
  const c3: uint8[] = [1]
  const c5: float32[] = [0.00600000005]
  const c6: uint8[] = [0]
  const c7: float32[] = [3.00003988e-41]
  const c8: uint8[] = [0]
  t9 = com.microsoft.QLinearMul(a1, c2, c3, b4, c5, c6, c7, c8)
  outputs: t9
  ```
- `mm-qlinearconv-dab856` (41 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 0, 0, 0] got **-128**, exact **0**, allowed slack 0
  ```
  input x1: int8[1, 64, 3, 2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[] = [1]
  const c3: int8[] = [0]
  const w4: int8[64, 16, 2, 2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c5: float32[64] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ...]
  const c6: int8[64] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c7: float32[] = [5.60519386e-45]
  const c8: int8[] = [0]
  t9 = QLinearConv(x1, c2, c3, w4, c5, c6, c7, c8) {'kernel_shape': [2, 2], 'strides': [3, 1], 'dilations': [2, 1], 'group': 4}
  outputs: t9
  ```
- `mm-matmul-ef058a` (40 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **0**, exact **255**, allowed slack 0
  ```
  input a1: uint8[1, 1] = [0]
  const w2: int8[1, 1] = [0]
  const c3: float32[] = [0.00800000038]
  const c4: uint8[] = [238]
  const c6: float32[] = [7.00649232e-45]
  const c7: int8[] = [94]
  const c10: float32[] = [1.40129846e-45]
  const c11: uint8[] = [0]
  t5 = DequantizeLinear(a1, c3, c4)
  t8 = DequantizeLinear(w2, c6, c7)
  t9 = MatMul(t5, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `obo-qlinearmul-b4fab1` (39 raw hits; fused at extended: `com.microsoft.QLinearMul`)
  - example: index [0, 0, 0] got **-39**, exact **-40**, allowed slack 0
  ```
  input a1: int8[1, 1, 1] = [0]
  const c2: float32[] = [2.99999784e-40]
  const c3: int8[] = [103]
  const c4: int8[1] = [0]
  const c5: float32[] = [9.99999975e-05]
  const c6: int8[] = [-128]
  const c7: float32[] = [9.99966584e-42]
  const c8: int8[] = [0]
  t9 = com.microsoft.QLinearMul(a1, c2, c3, c4, c5, c6, c7, c8)
  outputs: t9
  ```
- `obo-qlinearmatmul-31f3be` (36 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **0**, exact **1**, allowed slack 0
  ```
  input a1: uint8[1, 1] = [0]
  input b4: uint8[1, 1] = [0]
  const c2: float32[] = [3.0000001e+19]
  const c3: uint8[] = [0]
  const c5: float32[1] = [7.0000002e+28]
  const c6: uint8[1] = [0]
  const c7: float32[] = [1]
  const c8: uint8[] = [1]
  t9 = QLinearMatMul(a1, c2, c3, b4, c5, c6, c7, c8)
  outputs: t9
  ```
- `mm-conv-5acfa0` (35 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 31, 0, 0] got **255**, exact **0**, allowed slack 7
  ```
  input x1: uint8[1, 32, 1, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[] = [1]
  const c3: uint8[] = [0]
  const w5: uint8[32, 1, 1, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c6: float32[] = [7.00649232e-43]
  const c7: uint8[] = [0]
  const c9: int32[32] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c10: float32[] = [0]
  const c13: float32[] = [1.40129846e-45]
  const c14: uint8[] = [0]
  t4 = DequantizeLinear(x1, c2, c3)
  t8 = DequantizeLinear(w5, c6, c7)
  t11 = DequantizeLinear(c9, c10)
  t12 = Conv(t4, t8, t11) {'kernel_shape': [1, 1], 'group': 32}
  t15 = QuantizeLinear(t12, c13, c14)
  outputs: t15
  ```
- `obo-qlinearconv-6c81e2` (31 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 1, 0] got **45**, exact **46**, allowed slack 0
  ```
  input x1: uint8[1, 32, 1] = [144, 155, 142, 193, 177, 197, 0, 24, 1, 151, 19, 13 ...]
  const c2: float32[] = [8.99633614e-43]
  const c3: uint8[] = [128]
  const w4: uint8[8, 16, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c5: float32[8] = [1, 0.140000001, 1, 1, 1, 1, 1, 1]
  const c6: uint8[8] = [0, 234, 0, 0, 0, 0, 0, 0]
  const c7: float32[] = [9.96323208e-43]
  const c8: uint8[] = [139]
  const c9: int32[8] = [0, 4192, 0, 0, 0, 0, 0, 0]
  t10 = QLinearConv(x1, c2, c3, w4, c5, c6, c7, c8, c9) {'kernel_shape': [1], 'strides': [2], 'group': 2}
  outputs: t10
  ```
- `obo-qlinearconv-a5e2a9` (15 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 18, 0, 0] got **-19**, exact **-20**, allowed slack 0
  ```
  input x1: int8[1, 32, 1, 9] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[] = [0.00173000002]
  const c3: int8[] = [0]
  const w4: int8[32, 1, 2, 5] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c5: float32[] = [1.95999616e-41]
  const c6: int8[] = [0]
  const c7: float32[] = [1.89175293e-43]
  const c8: int8[] = [0]
  const c9: int32[32] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  t10 = QLinearConv(x1, c2, c3, w4, c5, c6, c7, c8, c9) {'kernel_shape': [2, 5], 'strides': [2, 3], 'dilations': [1, 2], 'pads': [1, 0, 1, 0], 'group': 32}
  outputs: t10
  ```
- `obo-matmul-7b6d61` (12 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **147**, exact **146**, allowed slack 0
  ```
  input a1: uint8[1, 1] = [190]
  const w2: uint8[1, 1] = [80]
  const c3: float32[] = [0.000577999977]
  const c4: uint8[] = [128]
  const c6: float32[1] = [5.00000108e-40]
  const c7: uint8[1] = [128]
  const c10: float32[] = [8.00001293e-42]
  const c11: uint8[] = [254]
  t5 = DequantizeLinear(a1, c3, c4)
  t8 = DequantizeLinear(w2, c6, c7) {'axis': 1}
  t9 = MatMul(t5, t8)
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `obo-qlinearmatmul-108f5b` (10 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **115**, exact **114**, allowed slack 0
  ```
  input a1: int8[1, 5] = [-128, 94, 0, 0, 127]
  const c2: float32[] = [0.00400000019]
  const c3: int8[] = [-56]
  const w4: int8[5, 1] = [51, 108, 0, 0, -65]
  const c5: float32[1] = [9.00054004e-42]
  const c6: int8[1] = [0]
  const c7: float32[] = [2.0038568e-43]
  const c8: int8[] = [0]
  t9 = QLinearMatMul(a1, c2, c3, w4, c5, c6, c7, c8)
  outputs: t9
  ```
- `obo-conv-8de96b` (9 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 0, 0, 0] got **2**, exact **3**, allowed slack 0
  ```
  input x1: uint8[1, 1, 1, 3] = [139, 25, 255]
  const c2: float32[] = [2.59996917e-41]
  const c3: uint8[] = [255]
  const w5: int8[1, 1, 1, 3] = [10, -95, 0]
  const c6: float32[1] = [0.000456999987]
  const c7: int8[1] = [13]
  const c10: float32[] = [1.13999834e-40]
  const c11: uint8[] = [0]
  t4 = DequantizeLinear(x1, c2, c3)
  t8 = DequantizeLinear(w5, c6, c7) {'axis': 0}
  t9 = Conv(t4, t8) {'kernel_shape': [1, 3], 'strides': [3, 1]}
  t12 = QuantizeLinear(t9, c10, c11)
  outputs: t12
  ```
- `mm-matmul-6a9f21` (1 raw hits; fused at extended: `QLinearMatMul`)
  - example: index [0, 0] got **0**, exact **114**, allowed slack 0
  ```
  input t12: uint8[1, 1] = [0]
  const c13: float32[] = [1]
  const c14: uint8[] = [0]
  const c16: uint8[1, 1] = [0]
  const c17: float32[] = [1]
  const c18: uint8[] = [0]
  const c21: float32[] = [9.99966584e-42]
  const c22: uint8[] = [114]
  t15 = DequantizeLinear(t12, c13, c14)
  t19 = DequantizeLinear(c16, c17, c18)
  t20 = MatMul(t15, t19)
  t23 = QuantizeLinear(t20, c21, c22)
  outputs: t23
  ```
- `obo-conv-relu-7a6628` (1 raw hits; fused at extended: `QLinearConv`)
  - example: index [0, 0, 0, 0] got **114**, exact **113**, allowed slack 0
  ```
  input x1: uint8[1, 16, 1, 3] = [146, 0, 242, 174, 0, 199, 179, 0, 160, 119, 0, 99 ...]
  const c2: float32[] = [0.015625]
  const c3: uint8[] = [128]
  const w5: int8[4, 8, 1, 3] = [0, 19, -22, 0, 4, -48, 0, -38, 115, 0, -117, 107 ...]
  const c6: float32[] = [5.50738322e-41]
  const c7: int8[] = [0]
  const c9: int32[4] = [3302, 0, 0, 0]
  const c10: float32[] = [8.60397257e-43]
  const c14: float32[] = [5.60519386e-43]
  const c15: uint8[] = [0]
  t4 = DequantizeLinear(x1, c2, c3)
  t8 = DequantizeLinear(w5, c6, c7)
  t11 = DequantizeLinear(c9, c10)
  t12 = Conv(t4, t8, t11) {'kernel_shape': [1, 3], 'strides': [1, 2], 'dilations': [1, 2], 'pads': [0, 2, 0, 2], 'group': 2}
  t13 = Relu(t12)
  t16 = QuantizeLinear(t13, c14, c15)
  outputs: t16
  ```

### RC7: valid models rejected (type combinations / per-row / per-channel zero points / blocked-with-1-block)

**Verdict:** Valid model rejected (unsupported type/feature).  
**Known upstream?** Per-row MatMulInteger zero point: [#27897](https://github.com/microsoft/onnxruntime/issues/27897) (closed, still rejected in 1.30); per-channel QLinearConv zero points: [#28447](https://github.com/microsoft/onnxruntime/issues/28447); QLinearMatMul zero-point shape limits: [#15442](https://github.com/microsoft/onnxruntime/issues/15442) (open).  

ORT's CPU EP errors out on models the ONNX checker accepts: QLinearMatMul/QLinearConv implement only (u8,u8,u8), (u8,s8,u8), (s8,s8,s8) of the 8 legal (a,b,y) type combos; MatMulInteger/QLinearMatMul reject per-row a zero points; ConvInteger rejects per-output-channel w zero points; Q/DQ with block_size>0 and exactly one block (scale shape [1]) is rejected as 'per-tensor with block_size'. These are loud failures (no wrong numbers) - conformance gaps, not correctness bugs.

- `rej-convinteger-f1891a` (1329 raw hits; fused at extended: `ConvInteger`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running ConvInteger node. Name:'n0_ConvInteger' Status Message: onnxruntime/core/providers/cpu/quantization/conv_integer.cc:52 virtu`
  ```
  input x1: int8[1, 1, 1, 3] = [0, 0, 0]
  const w2: int8[2, 1, 5, 2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c3: int8[] = [0]
  const c4: int8[2] = [0, 0]
  t5 = ConvInteger(x1, w2, c3, c4) {'kernel_shape': [5, 2], 'strides': [2, 1], 'dilations': [1, 2], 'pads': [3, 0, 3, 0]}
  outputs: t5
  ```
- `rej-qlinearmatmul-834f6d` (1146 raw hits; fused at extended: ``)
  - error: `[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for QLinearMatMul(21) node with name 'n0_QLinearMatMul'`
  ```
  input a1: uint8[1, 1] = [0]
  const c2: float32[] = [1]
  const c3: uint8[] = [0]
  const w4: int8[1, 1] = [0]
  const c5: float32[] = [1]
  const c6: int8[] = [0]
  const c7: float32[] = [1]
  const c8: int8[] = [0]
  t9 = QLinearMatMul(a1, c2, c3, w4, c5, c6, c7, c8)
  outputs: t9
  ```
- `rej-qlinearconv-e7eef0` (882 raw hits; fused at extended: ``)
  - error: `[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for QLinearConv(10) node with name 'n0_QLinearConv'`
  ```
  input x1: uint8[1, 1, 1, 1] = [0]
  const c2: float32[] = [1]
  const c3: uint8[] = [0]
  const w4: uint8[1, 1, 2, 1] = [0, 0]
  const c5: float32[1] = [1]
  const c6: uint8[1] = [0]
  const c7: float32[] = [1]
  const c8: int8[] = [0]
  const c9: int32[1] = [0]
  t10 = QLinearConv(x1, c2, c3, w4, c5, c6, c7, c8, c9) {'kernel_shape': [2, 1], 'strides': [3, 1], 'dilations': [1, 2], 'pads': [1, 0, 1, 0]}
  outputs: t10
  ```
- `rej-qlinearmatmul-65200e` (422 raw hits; fused at extended: `QLinearMatMul`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running QLinearMatMul node. Name:'n0_QLinearMatMul' Status Message: onnxruntime/core/providers/cpu/quantization/quantize_linear_matm`
  ```
  input a1: int8[2, 1] = [0, 0]
  const c2: float32[2] = [1, 1]
  const c3: int8[2] = [0, 0]
  const w4: int8[1, 1] = [0]
  const c5: float32[1] = [1]
  const c6: int8[1] = [0]
  const c7: float32[] = [1]
  const c8: int8[] = [0]
  t9 = QLinearMatMul(a1, c2, c3, w4, c5, c6, c7, c8)
  outputs: t9
  ```
- `rej-matmulinteger-76e216` (265 raw hits; fused at extended: `MatMulInteger`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running MatMulInteger node. Name:'n0_MatMulInteger' Status Message: onnxruntime/core/providers/cpu/quantization/matmul_integer.cc:63`
  ```
  input a1: int8[2, 1] = [0, 0]
  input b2: int8[1, 1] = [0]
  const c3: int8[2] = [0, 0]
  const c4: int8[] = [0]
  t5 = MatMulInteger(a1, b2, c3, c4)
  outputs: t5
  ```
- `rej-quantizelinear-286034` (191 raw hits; fused at extended: `QuantizeLinear`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running QuantizeLinear node. Name:'n0_QuantizeLinear' Status Message: onnxruntime/core/providers/cpu/quantization/quantize_linear.cc`
  ```
  input x1: float32[4] = [0, 0, 0, 0]
  const c2: float32[1] = [1]
  const c3: int8[1] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 4}
  outputs: t4
  ```
- `rej-dequantizelinear-872574` (161 raw hits; fused at extended: `DequantizeLinear`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running DequantizeLinear node. Name:'n0_DequantizeLinear' Status Message: onnxruntime/core/providers/cpu/quantization/quantize_linea`
  ```
  input x1: uint16[16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ...]
  const c2: float32[1] = [1]
  const c3: uint16[1] = [0]
  t4 = DequantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 32}
  outputs: t4
  ```
- `rej-quantizelinear-dequantizelinear-436772` (96 raw hits; fused at extended: `QuantizeLinear DequantizeLinear`)
  - error: `[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running QuantizeLinear node. Name:'n0_QuantizeLinear' Status Message: onnxruntime/core/providers/cpu/quantization/quantize_linear.cc`
  ```
  input x1: float32[3] = [0, 0, 0]
  const c2: float32[1] = [1]
  const c3: uint4[1] = [0]
  const obs_s5: float32[] = [1]
  const obs_z6: uint4[] = [0]
  t4 = QuantizeLinear(x1, c2, c3) {'axis': 0, 'block_size': 16}
  t7 = DequantizeLinear(t4, obs_s5, obs_z6)
  outputs: t7
  ```

