# qfuzz: a differential fuzzer and exact verifier for quantized ONNX Runtime

qfuzz generates random quantized ONNX graphs and runs each one through ONNX Runtime three times:
with graph optimizations off, at `ORT_ENABLE_EXTENDED` (where ORT fuses QDQ patterns into QLinear*, MatMulNBits
and similar kernels) and at `ORT_ENABLE_ALL`. It compares every result with an **exact reference interpreter** of
the ONNX spec. Any disagreement that falls outside what the spec allows is delta-debugged down to a few-element
graph and written out as a standalone `repro.py`.

The campaign described here ran 150,000 cases against onnxruntime 1.30.0 on a Xeon with AVX-512 VNNI and AMX.
It found **six root causes of wrong results** and one class of valid models that ORT rejects. Four of the
wrong-result causes reproduce with ordinary scales. Every one reproduces from a committed script, and the four
real bugs have paste-ready upstream reports in [`upstream/`](upstream/README.md).

An optional second backend, **Apache TVM 0.27** (Relax ONNX frontend, `llvm`), ran 20,000 cases with the same
oracle. It produced zero wrong results on the 16,360 cases it could compile and run. The remaining cases failed
loudly: one importer bug (int4 initializers), documented unsupported features, and undefined zero scales. See
[Second backend: Apache TVM](#second-backend-apache-tvm-from-resultstvm_campaignjson).

## Why this is hard to copy

- **The reference is exact, and it has an error model.** Integer paths use exact int64 arithmetic. Every
  `round()` is applied to the exact rational value with round-half-to-even. When the float64 approximation sits near
  a tie, the element is recomputed with `fractions.Fraction`. On top of that, every tensor carries a propagated
  **slack** bound. The slack is the largest deviation a conforming float32 implementation may legitimately show,
  which covers ties that float32 rounding can flip, requantization multipliers folded in float32, accumulation
  order, the subnormal range, overflow near FLT_MAX and int32 accumulator overflow. Far from ties the slack is 0,
  so integer outputs must match bit for bit. The slack is what lets qfuzz report "not a bug, rounding latitude"
  for the 11,889 cases where ORT and the ideal result differ, and still flag an off-by-one that sits nowhere near a
  tie.
- **Delta debugging on a JSON graph IR with dimension labels.** The minimizer shrinks coupled dimensions together
  (MatMul's K, Conv's C/M, whole quantization blocks) and never trades a realistic-scale bug for a
  degenerate-scale one, because the scale regime is part of the signature. It also cuts upstream subgraphs by
  substituting reference values. Most findings end up as a 1 to 4 node graph with 5 to 20 numbers (a few conv and MatMulNBits cases keep a few hundred).
- **Honest triage.** Four bugs in qfuzz's own reference were found and fixed during triage (listed below). Findings
  that need scales of 1e-40 or 1e30 are separated from findings that occur with realistic scales.

## Quickstart

```bash
cd qfuzz
pip install -e '.[test]'          # numpy, onnx, onnxruntime, ml_dtypes (+ pytest)
pytest -q                          # ~15 s: reference unit tests, generator validity, oracles, minimizer, all repros

qfuzz show 413 --minimize          # generate/judge/minimize one seed
qfuzz run --cases 150000 --workers 4 --time-limit 840   # -> results/campaign.json   (~6 min here)
qfuzz minimize --budget 60                               # -> findings/<id>/{repro.py,finding.json}
python scripts/make_triage.py                            # route findings to hand-written root causes
qfuzz report                                             # -> results/REPORT.md
python findings/<id>/repro.py --check                    # exit 1 if the discrepancy reproduces
python upstream/rc1_double_qdq_remover.py                # paste-ready upstream repro (AssertionError = bug present)
```

## Layout

| file | purpose |
|---|---|
| `src/qfuzz/case.py` | JSON-serialisable graph IR (nodes, tensors with dimension labels) -> ONNX model / feeds |
| `src/qfuzz/generator.py` | 16 patterns (see below) and edge-case samplers for scales, zero points and values |
| `src/qfuzz/reference.py` | exact interpreter with slack propagation (the oracle) |
| `src/qfuzz/runner.py`, `oracle.py` | ORT at 3 optimization levels; compare and classify; signature and scale regime |
| `src/qfuzz/minimize.py` | delta debugger (cut, trim, shrink labeled dims and blocks, ddmin values, round floats) |
| `src/qfuzz/repro.py` | standalone repro script emitter (numpy + onnx + onnxruntime only) |
| `src/qfuzz/campaign.py` | crash-isolated subprocess workers (a segfault is recorded and the worker restarts), dedupe; `--backend ort\|tvm` |
| `src/qfuzz/tvm_runner.py` | optional Apache TVM backend (Relax ONNX import, `llvm` build, Relax VM); error-stage classification |
| `results/campaign.json`, `results/REPORT.md`, `results/triage.json` | campaign output, report and curated triage |
| `results/tvm_campaign.json`, `results/TVM_REPORT.md`, `results/tvm_triage.json`, `results/tvm_on_ort_findings.json` | TVM campaign, report, triage (`scripts/make_tvm_triage.py`) and ORT findings replayed on TVM |
| `findings/<id>/` | minimized `repro.py` + `finding.json` for each of the 55 distinct findings; `ISSUE.md` drafts for the two best (superseded by `upstream/`) |
| `upstream/` | paste-ready bug reports (`*.md`) with standalone repro scripts (`*.py`); see `upstream/README.md` |

**Ops and patterns covered.**
- QuantizeLinear and DequantizeLinear: per-tensor, per-axis and blocked, in uint8, int8, uint16, int16, int4,
  uint4, float8e4m3fn and float8e5m2, with saturate 0 or 1.
- QLinearMatMul, QLinearConv, MatMulInteger, ConvInteger and DynamicQuantizeLinear, plus com.microsoft
  QLinearAdd and QLinearMul.
- QDQ sandwiches that ORT fuses: DQ->MatMul->Q, DQ->Conv(+bias, +Relu)->Q, DQ->Add/Mul->Q, DQ->Relu->Q,
  DynamicQuantizeLinear->MatMulInteger->Cast->Mul (DynamicQuantizeMatMul), and DQ(int4 blocked)->MatMul
  (MatMulNBits).
- Q->DQ->Q->DQ chains and two-stage MatMul chains.
- Edge cases: exact ties, saturation boundaries, boundary zero points, tiny, subnormal and huge scales,
  ±inf and ±3e38 inputs, and shapes at SIMD and blocking edges (15/16/17, 63/64/65, ..., 512).

## Campaign results (onnxruntime 1.30.0, from `results/campaign.json`)

150,000 cases, 4 workers, 350 s. Every case ran at 3 optimization levels. No crashes.

| verdict | cases | share |
|---|---:|---:|
| exact (bit-identical to the ideal result at all levels) | 124,525 | 83.0% |
| within-bound (differs, but inside spec latitude) | 11,889 | 7.9% |
| **mismatch** (beyond bound by ≥ 2 LSB or beyond the float bound) | 8,279 | 5.5% |
| **off-by-one** (1 LSB beyond bound, no rounding tie nearby) | 296 | 0.2% |
| **nan-inf** | 519 | 0.35% |
| **rejects-valid-model** (passes `onnx.checker` full_check, ORT errors) | 4,492 | 3.0% |

Oracle (a): in 9,727 cases (6.5%) fusion changed at least one output bit relative to `ORT_DISABLE_ALL`.

**Clean results.** Every case in these areas was exact or within bound:
- MatMulInteger (8,895 exact) and ConvInteger on this VNNI/AMX CPU
- DynamicQuantizeLinear and the fused DynamicQuantizeMatMul
- DQ->Relu->Q
- DequantizeLinear in every type (int32 is within bound rather than bit-exact; see below)
- two-stage chains, except one degenerate-scale case

The 55 raw signatures minimize to 55 findings, grouped below into 7 root causes. "hits" counts the campaign
cases in each group, split into realistic scales (inside [1e-12, 1e12]) and extreme scales.

| root cause | verdict | findings | hits (normal / extreme) |
|---|---|---:|---:|
| RC1 DoubleQDQPairsRemover re-quantizes Q->DQ->Q->DQ onto a new grid | **real ORT bug** (fusion changes results) | 5 | 568 / 85 |
| RC2 DQ(int4)->MatMul fused into MatMulNBits with int8 activations | intentional, documented trade-off; not spec-conformant | 3 | 5,198 / 990 |
| RC3 float8 QuantizeLinear with saturate=0 near FLT_MAX | **real ORT bug** | 2 | 300 / 193 |
| RC4 blocked (and 4-bit per-axis) QuantizeLinear saturates to the wrong end at ≥ 2^31 / +inf | **real ORT bug** | 10 | 94 / 340 |
| RC5 QLinearAdd/QLinearMul (and fused DQ->Add/Mul->Q) saturate to the wrong end when out of int32 range | **real ORT bug**, mostly needs extreme scale ratios | 8 | 8 / 524 |
| RC6 requant multiplier folded in float32 breaks with subnormal or huge scales | real, but **degenerate inputs only** | 19 | 0 / 794 |
| RC7 valid models rejected (type combos, per-row or per-channel zero points, single-block) | conformance gap (loud error, no wrong numbers) | 8 | 4,492 |

`results/REPORT.md` lists every finding with its minimized graph. **Paste-ready upstream reports** for RC1, RC3, RC4
and RC5 (plus one for the matching `onnx.reference` bug) are in [`upstream/`](upstream/README.md). Each follows the
onnxruntime bug-report form, has a ≤40-line inline repro with its real output, quotes the spec text, points at the
offending source lines and lists the related issues. **Nothing has been filed.**

### Findings with triage

**RC1: DoubleQDQPairsRemover (real bug, realistic).**
- Repro: int16, `x=-6.504 -> Q(s=1) -> DQ -> Q(s=0.001) -> DQ`. The spec and `ORT_DISABLE_ALL` give -7.0;
  `ORT_ENABLE_EXTENDED` gives -6.504.
- Cause: `double_qdq_pairs_remover.cc` replaces both pairs with one pair whose scale is the intersected range
  divided by (qmax-qmin). That grid is neither s1 nor s2, so the coarse rounding step disappears. Another example:
  uint16 with z2 != z1 turns 20000 into 19999.8. With huge scales the new scale overflows to inf and a 0 input
  becomes NaN.
- The spec defines the graph as the composition of the four ops, so this is a correctness bug in an optimization
  that is on by default.
- The transformer is registered at Level 1, so `ORT_ENABLE_BASIC` is already affected (source confirmed in
  `FindNewZeroPointAndScale`). The workaround is `session.disable_double_qdq_remover=1`.
- Upstream: no matching issue found. Related: #32132 (fused QLinear ops off by k steps) and #28030.

**RC2: MatMulNBits accuracy level (by design).**
- At EXTENDED and ALL, DQ(int4/uint4, blocked)->MatMul becomes `com.microsoft.MatMulNBits` with
  `accuracy_level=4`, which quantizes the float activations to int8. The unoptimized graph computes a float32
  MatMul.
- Deviations: in 87 sampled realistic-scale cases, the largest error per case, relative to the largest output
  magnitude, has a median of 0.6%, a 90th percentile of 0.9% and a maximum of 6.5%. That is far beyond float32
  rounding (about 1e-6).
- `session.qdq_matmulnbits_accuracy_level=0` removes every deviation (verified), so this is a documented knob with
  a surprising default rather than a kernel bug.
- Upstream: its LLM impact is tracked in #29849.

**RC3: float8 saturate=0 (real bug).**
- The Cast table defines `[x] > FLT_MAX` as NaN for E4M3FN and ±Inf for E5M2, where `[x]` is x rounded to the
  mantissa width. onnx.reference agrees.
- ORT maps |x| in [480, 496) to ±448 for E4M3FN, even though 479 and 496 map to NaN (non-monotonic).
- ORT maps (61440, 65536) to NaN instead of ±Inf for E5M2.
- Found by exhaustive sweep after the fuzzer hit it. Same result for per-tensor, per-axis and blocked, and for `Cast`.
- Mechanism (from the source, `include/onnxruntime/core/common/float8.h`): for E4M3FN an unconditional
  `val &= 0xFE` maps the NaN code to 448. For E5M2, `val |= 0x7C` ORs the Inf pattern into the mantissa 0x7B, which
  gives NaN.
- Upstream: no matching issue found.

**RC4: blocked QuantizeLinear sign flip (real bug).**
- Example: uint8, blocked along a non-innermost axis, `x=+inf` with scale 1 gives **0** instead of 255.
- Any |x/scale| ≥ 2^31 fails, and a negative zero point turns -3e38 into +7 for int4.
- Probing layouts: 8/16-bit types fail only when the blocked axis is not innermost. int4/uint4 also fail for
  per-axis quantization of the innermost axis. Per-tensor is always correct.
- Mechanism (confirmed in the source, `onnxruntime/core/util/qmath.h`): `static_cast<int32_t>(std::nearbyint(x / sc))`
  runs *before* `std::clamp`. That is undefined for out-of-range values, and x86 gives INT32_MIN. The per-tensor path
  uses `MlasQuantizeLinear`, which clamps in float.
- `onnx.reference` has the same defect (`np.rint(x).astype(np.int32)` before `np.clip`), so it cannot serve as the
  oracle here. See `upstream/onnx_reference_quantize_overflow.md`.
- Upstream: no matching issue found.

**RC5: QLinearAdd/QLinearMul overflow (real bug, low likelihood).**
- Example: uint8 with a=229, a_scale=3000, c_scale=1e-4 gives an exact value of 6.9e9, which should saturate to
  255. The fused QLinearAdd returns 0. The unfused graph is correct.
- Only 8 of 532 hits used scales inside [1e-12, 1e12].
- Mechanism (from the source): the x86 MLAS kernels (`qladd_avx2.cpp`, `qladd.cpp`, `qlmul.cpp`) convert with
  `cvtps2dq` and saturate only afterwards with `packs`/`packus`.

**RC6: extreme scales (real, degenerate).**
- ORT folds `a_scale*b_scale/y_scale` into one float32 value. That value underflows to subnormal or overflows to
  inf. Results are off by 1–3 LSB far from any tie, and acc=0 can give qmin (0*inf=NaN) instead of the zero point.
- The spec's exact result is well defined, so these are real non-conformances, but no real quantizer emits such
  scales. Kept separate on purpose.

**RC7: rejected valid models (conformance gaps).**
- QLinearMatMul and QLinearConv implement 3 of the 8 legal (a, b, y) type combinations.
- MatMulInteger and QLinearMatMul reject per-row `a` zero points (#27897 is closed, but 1.30 still rejects them).
- ConvInteger rejects per-channel `w_zero_point`.
- Q/DQ with `block_size>0` and a one-block scale of shape `[1]` is rejected as "per-tensor".

## Second backend: Apache TVM (from `results/tvm_campaign.json`)

`src/qfuzz/tvm_runner.py` runs a case through **Apache TVM 0.27.0**. It imports the model with the Relax
ONNX frontend (`from_onnx`, with Relay removed upstream), applies `DecomposeOpsForInference` and `LegalizeOps`,
compiles with `tvm.compile(target="llvm")` and executes on the Relax VM. The result goes through the same exact
reference and slack oracle as ORT. TVM is optional (`pip install -e '.[tvm]'`, which pulls the `apache-tvm` wheel,
`py3-none-manylinux_2_28_x86_64`, about 9 s to install here). Tests that need TVM are skipped without it.
Errors are classified by stage: `tvm-unsupported` (the importer refuses a feature explicitly) or
`tvm-error-import`, `tvm-error-build` or `tvm-error-run` (anything else).

```bash
pip install -e '.[tvm]'
qfuzz run --backend tvm --cases 20000 --workers 2 --out results/tvm_campaign.json
python scripts/make_tvm_triage.py        # -> results/tvm_triage.json, results/TVM_REPORT.md
qfuzz show 1455 --pattern dq_matmul_q --backend tvm
```

The importer has no converter for QLinearMatMul, QLinearConv, ConvInteger or com.microsoft QLinearAdd/Mul, so the
TVM campaign cycles evenly through the 12 generator patterns built only from ops it converts: Q, DQ, Q->DQ chains,
DQ->MatMul/Conv/Add/Mul/Relu->Q, MatMulInteger, DynamicQuantizeLinear (+MatMulInteger), DQ(int4)->MatMul and
two-stage chains.

20,000 cases, 2 workers, 2028 s, on the same machine and ORT/onnx versions as above:

| verdict | cases | share |
|---|---:|---:|
| exact | 15,055 | 75.3% |
| within-bound | 1,305 | 6.5% |
| tvm-unsupported | 1,333 | 6.7% |
| tvm-error-import | 2,219 | 11.1% |
| tvm-error-build | 88 | 0.4% |

**Zero findings.** All 16,360 cases that TVM compiled and ran were bit-exact or inside spec latitude: no mismatch,
off-by-one, nan-inf or crash. This covers inf and ±3e38 inputs, exact ties, boundary zero points and tiny or huge
scales, the same edge cases that expose RC4 to RC6 in ORT. TVM's output is bit-identical to ORT `ORT_DISABLE_ALL`
in 16,069 of those cases. In the other 257 they differ, and TVM's result is inside spec latitude in every one.
A fault-injection test (`tests/test_tvm.py`) checks that the TVM path really goes
through the oracle.

The 3,640 cases that did not run split into three hand-triaged groups (`results/tvm_triage.json`):

| group | verdict | cases |
|---|---|---:|
| T1: any int4/uint4 initializer fails to import (TensorCopyFromBytes size mismatch) | tvm-importer-bug | 2,219 |
| T2: features the importer rejects explicitly (not bugs) | unsupported | 1,333 |
| T3: y_scale = 0 fails at compile time (Divide by zero in constant folding) | spec-undefined | 88 |

T1 is a real importer bug (a loud failure on valid models, not wrong numbers). It also blocks every int4-weight
graph, so TVM's handling of blocked int4 (ORT's RC2 area) is **untested**. No matching apache/tvm issue was found,
and no upstream report is drafted for it yet. T2 lists documented limitations. T3 is undefined input.

**ORT's 55 minimized findings replayed on TVM** (`results/tvm_on_ort_findings.json`):

| ORT root cause | TVM verdict per finding |
|---|---|
| RC1-double-qdq-remover | exact: 5 |
| RC2-matmulnbits-accuracy-level | tvm-error-import: 3 |
| RC3-float8-saturate-false | tvm-unsupported: 2 |
| RC4-blocked-q-int32-overflow | tvm-unsupported: 10 |
| RC5-qlinear-binary-overflow | exact: 7, tvm-unsupported: 1 |
| RC6-requant-float32-extreme-scales | exact: 8, tvm-unsupported: 11 |
| RC7-rejects-valid | exact: 1, tvm-unsupported: 7 |

TVM gets the spec answer on every RC1 and RC5 case it can import, on 8 of the RC6 degenerate-scale cases and on
the one RC7 model it accepts. It cannot express RC3 (float8 `saturate=0`) or RC4 (blocked Q) at all. The RC1 and RC5
upstream reports cite TVM as an independent second implementation.

## What is real and what is not

- **Real, verified by hand against the spec text and the minimized repro:** RC1, RC3 and RC4 (realistic trigger:
  an `inf` activation), plus RC5 in its rarer realistic-scale form. For each, the spec formula agrees with qfuzz
  and disagrees with ORT. So does `ORT_DISABLE_ALL` for RC1 and RC5, and ORT's own per-tensor kernel for RC4.
  `onnx.reference` agrees for RC1 and RC3. For RC4 and RC5 it shares the bug: it casts to int32 before clipping.
  That is reported separately in `upstream/onnx_reference_quantize_overflow.md`, and it is why the upstream reports
  do not lean on it there. TVM 0.27 independently returns the spec values for RC1 and RC5.
  The offending source lines were read for all four (see `upstream/*.md`), and the relevant files are unchanged on
  ORT `main` as of 2026-09-29. The ORT nightly itself could not be installed from the sandbox, so none of the
  bugs has been re-run on a nightly build.
- **Real but degenerate:** RC6, plus most RC5 and RC4 hits. They need scales of 1e-40 to 1e30 or inputs of about
  3e38. They are listed so they don't hide in the noise, not because they matter in practice.
- **Not a bug:**
  - RC2 is an intentional accuracy trade-off.
  - The 11,889 within-bound cases are spec latitude. Example: the spec's own DynamicQuantizeLinear example gives
    179 for x=0.5 only because float32 division rounds 25.4999991 up to 25.5; the exact answer is 178, and qfuzz
    accepts both.
  - Of the 9,727 cases where fusion changed bits, 2,222 are within latitude (the fused kernel lands on the other
    side of a rounding tie) and 7,504 are the findings above, mostly RC2.
  - DequantizeLinear of int32 values above 2^24 is not bit-exact: ORT converts int32 to float before
    subtracting and multiplying, which double-rounds. qfuzz accepts this within its 2-ulp latitude.
- **TVM:** the zero-findings result covers only the op subset TVM imports (T1 and T2 above), on the `llvm` CPU
  target with the default Relax pipeline. T1 (int4 initializers) is a real but loud TVM importer bug. It is
  triaged from reading the importer source, and no upstream report has been drafted. T3 (zero scale) is undefined
  input, not a bug. The TVM findings path was checked with fault injection, but no TVM case needed the minimizer.
- **Bugs in qfuzz's own reference, found during triage and fixed.** Each of these produced false positives until
  it was fixed:
  1. The float slack lacked an absolute subnormal term, so a subnormal MatMul output was flagged.
  2. float8 RNE at the overflow tie: 464 → 448 for E4M3FN.
  3. A finite ORT result next to an ideal value that overflowed to inf was reported as nan-inf, even when a 1-LSB
     latitude upstream permits it.
  4. Zero scales and int32 accumulator overflow were treated as defined. They are now "undefined" (infinite
     slack). ORT wraps int32 bias overflow in QLinearConv, which the QLinearMatMul spec text ("accumulation may
     overflow if and only if in 32 bits") arguably allows.
- **Limits of the oracle.**
  - The slack model assumes float32 intermediates with at most about 16 ulp of error per requantization. A kernel
    that is *legitimately* less precise, such as a fixed-point requant with a 1-LSB error away from ties, would be
    reported as off-by-one. None of the off-by-one findings here needed that excuse: all are RC6 degenerate-scale
    cases.
  - The reference does not model FTZ/DAZ.
  - ORT results can depend on the ISA. MLAS picks VNNI, AMX or AVX2 kernels at runtime, and u8s8 GEMMs on AVX2
    without VNNI use `vpmaddubsw` (int16 saturation). This machine never takes that path, so qfuzz saw zero
    MatMulInteger mismatches here. On other CPUs the repros may not reproduce (the tests skip in that case) and a
    campaign may find more.

## Spec ambiguities and the interpretation taken

| topic | spec text | qfuzz interpretation |
|---|---|---|
| precision of `x / y_scale` | QuantizeLinear: "type of `y_scale` determines the precision of the division" (newer text); older opsets are silent | ideal = exact rational; slack of 1 LSB where float32 division/reciprocal-multiply could cross a tie |
| requantization in QLinearMatMul/Conv | formula only, no intermediate precision | ideal = exact `acc*a_s*b_s/y_s`; slack 2^-20 relative (float32 folding); scales beyond float32 range are *not* excused (reported as RC6) |
| int32 accumulation overflow | "accumulation may overflow if and only if in 32 bits" | undefined → any result accepted |
| DynamicQuantizeLinear on all-zero input | formula divides by zero | undefined (ORT returns scale 1.0; onnx.reference returns 1/255) |
| zero scale | not forbidden, division by zero | undefined |
| float8 `[x] > FLT_MAX` with saturate=0 | `[x]` = "value rounded to the target mantissa width" | RNE with unbounded exponent, then compare; ties at the boundary follow the even encoding (matches onnx.reference) |
| QLinearAdd / QLinearMul (com.microsoft) | no formal spec | exact real arithmetic + round-half-to-even, like QuantizeLinear |
| ConvInteger/QLinearConv padding | not stated | pad with the zero point (padding contributes 0 after dequantization) |
| NaN into integer QuantizeLinear | not stated | undefined (not generated) |

## Tests and CI

`pytest -q` (about 20 s, 122 tests; the TVM tests are skipped when `tvm` is not installed) covers:
- reference unit tests against hand-computed and spec-example values
- validity of every generator pattern under `onnx.checker` full_check
- oracle plumbing, including injected mismatches
- the minimizer on a synthetic injected bug (it must shrink to at most 4 elements)
- every `findings/*/repro.py`: it must run, and it is skipped if the discrepancy does not reproduce on the local
  ORT/ISA, so CI cannot flake across CPUs
- the report CLI
- every `upstream/*.py` script: it must end with `AssertionError: <message>` (bug present) or exit 0 (skipped as
  "does not reproduce"). The script must be ≤ 40 lines and import only numpy/onnx/onnxruntime, and its `.md` must
  embed the current copy
- the TVM backend: error classification, summary fields and committed results, which run without TVM; plus spec
  ties and saturation, the scalar-rank regression, the unsupported and int4 classification and a fault-injection
  check, which run with TVM

CI should run:

```bash
cd qfuzz && pip install -e '.[test]' && pytest -q
# optional, to also run the TVM tests:
pip install -e '.[test,tvm]' && pytest -q tests/test_tvm.py
```

## Next steps

- **Other compilers.** TVM is done (see above). Next are XLA and StableHLO `uniform_quantize`, TensorRT (explicit
  Q/DQ), OpenVINO and ORT's non-CPU EPs. The generator, reference and minimizer are backend-agnostic, and a
  backend is a `run(case) -> OrtResult` function plus an error classifier (see `tvm_runner.py`). Differential runs
  across EPs of the same ORT build are the cheapest next win.
- **TVM follow-ups.** Minimization is still ORT-only: `qfuzz minimize` re-judges with ORT, which did not matter
  here because TVM had no findings. Once T1 is fixed (or by feeding int4 weights as uint8 plus a shift), re-run
  `dq_matmul_nbits` on TVM. Also try non-`llvm` targets, and any QDQ-to-integer rewrite passes,
  which would be TVM's analogue of the ORT fusions where RC1, RC2 and RC5 live.
- **More operators.** QLinearAveragePool, QLinearSoftmax, QGemm, QAttention, MatMulNBits with explicit
  accuracy levels, and float16 scales (where `precision` matters).
- **More ISAs.** Pin MLAS ISA dispatch, or run on AVX2-only and ARM runners, to fuzz the `vpmaddubsw` u8s8
  saturation path.
- **File upstream after review.** The reports are ready in `upstream/`. File RC1 first, then RC4, RC3 and RC5, after
  re-running them on an ORT nightly (the nightly feed was not reachable from the sandbox that produced them).
