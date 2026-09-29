# Upstream bug reports (paste-ready, NOT filed)

Each `*.md` here is a complete bug report. Its sections follow the onnxruntime GitHub bug-report form
(Describe the issue / To reproduce / Urgency / Platform / OS Version / ONNX Runtime Installation / ONNX Runtime
Version or Commit ID / ONNX Runtime API / Architecture / Execution Provider / Execution Provider Library Version).
Each report embeds a short standalone script from this directory (onnx + onnxruntime + numpy only, ≤ 40 lines,
builds its model inline) and the real output of that script. Every script exits with
`AssertionError: <message>` while the bug is present and exits 0 once it is fixed. `tests/test_upstream.py` runs all
of them, skips any that do not reproduce on the runner, and checks that each `.md` embeds the current script.

Nothing in this directory has been posted anywhere.

| file | repo | bug | qfuzz root cause | realistic trigger? | related (not duplicates) |
|---|---|---|---|---|---|
| [`rc1_double_qdq_remover.md`](rc1_double_qdq_remover.md) | microsoft/onnxruntime | `DoubleQDQPairsRemover` re-quantizes Q->DQ->Q->DQ onto a new grid (on by default, from `ORT_ENABLE_BASIC`) | RC1 | **yes**: ordinary scales | #32132, #28030, #21319 |
| [`rc4_blocked_quantize_saturation.md`](rc4_blocked_quantize_saturation.md) | microsoft/onnxruntime | blocked (and int4/uint4 per-axis) QuantizeLinear: +inf and \|x/s\| ≥ 2^31 saturate to qmin | RC4 | yes, once an `inf` activation appears | #3442, PR #32452 |
| [`rc3_float8_saturate_false.md`](rc3_float8_saturate_false.md) | microsoft/onnxruntime | float8 `saturate=0`: E4M3FN [480, 496) gives 448 instead of NaN; E5M2 [61440, 65536) gives NaN instead of Inf (QuantizeLinear and Cast) | RC3 | only with `saturate=0` (non-default) | #16938 |
| [`rc5_qlinear_add_mul_saturation.md`](rc5_qlinear_add_mul_saturation.md) | microsoft/onnxruntime | fused QLinearAdd/QLinearMul (x86 MLAS) saturate to qmin when the result exceeds int32 | RC5 | rarely: needs a scale ratio above about 1e7 | #32132, PR #32754, #29727 |
| [`onnx_reference_quantize_overflow.md`](onnx_reference_quantize_overflow.md) | **onnx/onnx** | `onnx.reference` QuantizeLinear casts to int32 before clipping (the same class as RC4 and RC5) | found while writing RC4 and RC5 | same as RC4 | onnx#7835, onnx#8227 |

RC2 (MatMulNBits `accuracy_level=4` default) is by design, RC6 needs degenerate scales (1e-40, 1e30), and RC7 is
loud rejection of valid models. `results/triage.json` explains each. None of them gets a report here.

## Filing order

1. **RC1 first.** It is the clearest case. A *default* optimization level changes numbers for ordinary scales,
   `ORT_DISABLE_ALL`, `onnx.reference` and the spec formula all agree, the offending source lines are identified,
   and a one-line session-option workaround exists.
2. **RC4.** Sign-flipping saturation in the format used for LLM weight and activation quantization. The source
   lines are identified.
3. **RC3.** Precise spec-table violation with the exact bit-level cause in `float8.h`. It only matters with `saturate=0`.
4. **RC5.** The cause is clear, but it needs extreme scale ratios, so expect low priority.
5. The **onnx/onnx** reference report is independent and can go in any time. Filing it first actually helps RC4
   and RC5, because both reports explain why they do not use `onnx.reference` as the oracle.

## How to file

1. Re-run the script against the newest build you can get (next section). If it exits 0, the bug is fixed:
   don't file it.
2. Open https://github.com/microsoft/onnxruntime/issues/new/choose, pick **Bug Report**, and use the title from the
   quote block at the top of the report.
3. Paste each `###` section of the `.md` into the matching form field: Describe the issue, To reproduce, Urgency,
   and so on. The drop-down answers are given verbatim. Put the "Related issues" footer at the end of
   "Describe the issue".
4. Search the issue tracker once more for the title keywords right before filing. The duplicate searches here
   were done on 2026-09-29.
5. For the onnx/onnx report use https://github.com/onnx/onnx/issues/new/choose → Bug Report.

## Checking on the latest ORT / nightly: what was tried (2026-09-29)

- `pip install --pre onnxruntime`: PyPI has no pre-release newer than **1.30.0** (uploaded 2026-09-10), which is the
  version all outputs here were produced with (`git-commit-id=f2c39fe`).
- ORT nightly feed (`pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime`):
  **not reachable from this sandbox**, because the egress proxy denied the connection. The nightly was therefore
  *not* tested.
- Fallback: the relevant sources were compared at tag `v1.30.0` against `main` (fetched from raw.githubusercontent.com
  on 2026-09-29):

  | file | v1.30.0 vs main |
  |---|---|
  | `onnxruntime/core/optimizer/double_qdq_pairs_remover.cc` (RC1) | identical |
  | `onnxruntime/core/optimizer/graph_transformer_utils.cc` (RC1 registration, Level 1) | identical |
  | `include/onnxruntime/core/common/float8.h` (RC3) | identical |
  | `onnxruntime/core/providers/cpu/quantization/quantize_linear.cc` (RC4) | identical |
  | `onnxruntime/core/util/qmath.h` (RC4) | 5 added lines (an early return in the 2-bit helper); every cited conversion site unchanged |
  | `onnxruntime/core/mlas/lib/qladd.cpp`, `qlmul.cpp`, `contrib_ops/cpu/quantization/qlinear_binary_op.cc` (RC5) | identical |
  | `onnxruntime/core/mlas/lib/quantize.cpp` | a RISC-V `#if` only |

  So all four bugs are very likely still present on `main`. That is a source-level argument, not a run.

**Before filing, please confirm on a nightly if you can:**

```bash
python -m venv /tmp/ortnightly && . /tmp/ortnightly/bin/activate
pip install numpy onnx
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
cd upstream && for f in rc*.py; do python "$f" >/dev/null 2>&1; echo "$f exit=$? (1 = still reproduces)"; done
```

If a script still fails there, replace the version line in the report with the nightly version.
