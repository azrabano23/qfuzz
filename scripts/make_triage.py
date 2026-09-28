"""Assign each minimized finding to a hand-triaged root cause -> results/triage.json.

The root-cause texts below were written after reading the ONNX operator spec
(onnx 1.23 `onnx.defs` docs), the minimized repro and, where noted, the ORT
source.  The rules only route findings to those texts; every routed finding was
eyeballed (see README "What is real and what is not").  Re-run after
`qfuzz minimize`; unknown findings are left as "untriaged".
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RC = {
    "RC1-double-qdq-remover": dict(
        verdict="ort-bug",
        title="RC1: DoubleQDQPairsRemover re-quantizes Q1->DQ1->Q2->DQ2 onto a *new* grid (graph optimization changes results)",
        known="No matching issue found (searched 'DoubleQDQPairsRemover', 'double QDQ'); related but different: "
              "[#32132](https://github.com/microsoft/onnxruntime/issues/32132) (fused QLinear* off by k steps), "
              "[#28030](https://github.com/microsoft/onnxruntime/issues/28030) (double QDQ crash, closed).",
        summary="At ORT_ENABLE_EXTENDED/ALL the `DoubleQDQPairsRemover` replaces Q1(s1,z1)->DQ1->Q2(s2,z2)->DQ2 by a single "
                "Q->DQ pair whose scale is `(min(real_max1,real_max2)-max(real_min1,real_min2))/(qmax-qmin)` "
                "(onnxruntime/core/optimizer/double_qdq_pairs_remover.cc, FindNewZeroPointAndScale). That new grid is "
                "neither s1 nor s2, so values that the original graph snaps to the s1 grid (or to the coarser s2 grid) "
                "come out different: e.g. int16, x=-6.504, s1=1, s2=0.001 gives -7.0 unoptimized (spec) but -6.504 "
                "optimized; uint16 x=20000, s1=s2=1, z2=30458 gives 20000 vs 19999.8. When the two pairs have equal "
                "ranges but different zero points the result is off by up to several LSBs. The ONNX spec defines the "
                "graph as the composition of the four ops, so the optimized result is non-conforming. Realistic scales; "
                "extreme-scale variants additionally overflow the new-scale computation (inf/NaN outputs for x=0)."),
    "RC2-matmulnbits-accuracy-level": dict(
        verdict="by-design",
        title="RC2: DQ(int4 blocked)->MatMul is fused into MatMulNBits with int8-quantized activations by default",
        known="Behaviour is controlled by session option `session.qdq_matmulnbits_accuracy_level` (default 4). "
              "Accuracy impact on LLMs reported in [#29849](https://github.com/microsoft/onnxruntime/issues/29849) (open).",
        summary="At ORT_ENABLE_EXTENDED/ALL, DequantizeLinear(int4/uint4, blocked, axis=0) feeding a float MatMul is "
                "replaced by com.microsoft.MatMulNBits with accuracy_level=4, which quantizes the *float activations* to "
                "int8 per block. The unoptimized graph (and the spec) compute a float32 MatMul, so the optimized "
                "result deviates by far more than float32 rounding (87 sampled realistic-scale cases: largest error per "
                "case relative to the largest output magnitude has a median of 0.6% and a maximum of 6.5%). Setting `session.qdq_matmulnbits_accuracy_level` to 0 removes every deviation "
                "(verified). It is a documented speed/accuracy trade-off, but it is applied silently by a graph "
                "optimization level that is on by default - worth a clear note, not a kernel bug."),
    "RC3-float8-saturate-false": dict(
        verdict="ort-bug",
        title="RC3: QuantizeLinear to float8 with saturate=0 mis-handles the rounding band just above FLT_MAX",
        known="No matching issue found (searched 'float8 saturate false', 'E5M2 NaN instead of infinity'); "
              "[#16938](https://github.com/microsoft/onnxruntime/issues/16938) is a different float8 off-by-one.",
        summary="ONNX Cast/QuantizeLinear float8 table (saturate=False): `[x] > FLT_MAX -> NaN` for E4M3FN and "
                "`-> Inf` for E5M2, where [x] is x rounded (RNE) to the target mantissa. onnx.reference agrees. ORT "
                "1.30 CPU: E4M3FN maps |x/scale| in [480, 496) to +/-448 (a finite value) although 479 and 496 map to "
                "NaN - non-monotonic and contradicting the table; E5M2 maps (61440, 65536) to NaN instead of +/-Inf. "
                "Swept exhaustively in 0.25/16 steps; same for per-tensor, per-axis and blocked."),
    "RC4-blocked-q-int32-overflow": dict(
        verdict="ort-bug",
        title="RC4: blocked (and 4-bit) QuantizeLinear saturates to the WRONG end when abs(x/scale) >= 2^31 (incl. x=+inf)",
        known="No matching issue found (searched 'QuantizeLinear block_size saturation', 'int4 wrong saturation').",
        summary="Spec: y = saturate(round(x/scale)+zp). ORT returns the type *minimum* (or, with a negative zero "
                "point, the maximum) when |x/scale| >= 2^31 or x=+/-inf, e.g. uint8, blocked along a non-innermost "
                "axis, x=+inf -> 0 instead of 255; int4 blocked, x=-3e38, zp=-7 -> +7 instead of -8. Layout matters "
                "(probed with x=+inf, scale 1, zp 0): uint8/int8/int16/uint16 are wrong when the blocked axis is not "
                "the innermost one and right otherwise; int4/uint4 are also wrong for per-axis quantization of the "
                "innermost axis. Per-tensor is always right. Values below 2^31 are always right, which points at a "
                "float->int32 conversion before clamping (x86 cvtps2dq returns 0x80000000 for out-of-range input) in "
                "the strided/scalar code path - a hypothesis, not verified in the source. +inf activations (fp "
                "overflow upstream) make this reachable with ordinary scales."),
    "RC5-qlinear-binary-overflow": dict(
        verdict="ort-bug",
        title="RC5: QLinearAdd/QLinearMul (incl. fused DQ->Add/Mul->Q) saturate to the wrong end when the requantized value exceeds int32",
        known="No matching issue found (searched 'QLinearAdd QLinearMul saturation overflow').",
        summary="com.microsoft.QLinearAdd/QLinearMul (used by ORT's QDQ fusion of DQ->Add/Mul->Q at EXTENDED/ALL) "
                "return qmin instead of qmax (or vice versa) when (A*B or A+B)/C_scale is beyond ~2^31, e.g. uint8 "
                "a=229, a_scale=3000, c_scale=1e-4: exact 6.9e9 saturates to 255, ORT returns 0. The unfused graph is "
                "correct, so the fusion changes the result from 255 to 0. Requires a large ratio between input and "
                "output scales (badly calibrated model) - plausible but uncommon: only 8 of the 532 campaign hits used "
                "scales inside [1e-12, 1e12]; the rest overlap with the degenerate-scale regime of RC6."),
    "RC6-requant-float32-extreme-scales": dict(
        verdict="ort-bug-degenerate",
        title="RC6: requantization multiplier folded in float32 breaks for extreme / subnormal scales (QLinearMatMul, QLinearConv, QLinearMul, fused variants)",
        known="Not searched further - degenerate inputs; related saturation class: "
              "[#29727](https://github.com/microsoft/onnxruntime/issues/29727) (QLinearSoftmax saturation, closed).",
        summary="ORT folds a_scale*b_scale/y_scale into one float32 multiplier. With scales outside ~[1e-12,1e12] the "
                "product under/overflows (subnormal -> few mantissa bits, or inf -> 0*inf = NaN -> garbage int), "
                "producing off-by-one/two results far from any rounding tie, or even qmin for acc=0 (expected: the "
                "zero point). The spec defines the exact result, so these are real non-conformances, but only with "
                "scales no real quantizer emits. Reported for completeness; low priority."),
    "RC7-rejects-valid": dict(
        verdict="unsupported",
        title="RC7: valid models rejected (type combinations / per-row / per-channel zero points / blocked-with-1-block)",
        known="Per-row MatMulInteger zero point: [#27897](https://github.com/microsoft/onnxruntime/issues/27897) (closed, still rejected in 1.30); "
              "per-channel QLinearConv zero points: [#28447](https://github.com/microsoft/onnxruntime/issues/28447); "
              "QLinearMatMul zero-point shape limits: [#15442](https://github.com/microsoft/onnxruntime/issues/15442) (open).",
        summary="ORT's CPU EP errors out on models the ONNX checker accepts: QLinearMatMul/QLinearConv implement only "
                "(u8,u8,u8), (u8,s8,u8), (s8,s8,s8) of the 8 legal (a,b,y) type combos; MatMulInteger/QLinearMatMul "
                "reject per-row a zero points; ConvInteger rejects per-output-channel w zero points; Q/DQ with "
                "block_size>0 and exactly one block (scale shape [1]) is rejected as 'per-tensor with block_size'. "
                "These are loud failures (no wrong numbers) - conformance gaps, not correctness bugs."),
}


def route(f: dict, case: dict) -> str | None:
    ops = f["ops"]
    fused = f["fused_ops"].get("extended", [])
    dtypes = {t["dtype"] for t in case["inputs"] + case["inits"]}
    q_attrs = [n["attrs"] for n in case["nodes"] if n["op"] == "QuantizeLinear"]
    if f["cls"] == "rejects-valid-model":
        return "RC7-rejects-valid"
    if any(d.startswith("float8") for d in dtypes):
        return "RC3-float8-saturate-false"
    if "com.microsoft.MatMulNBits" in fused:
        return "RC2-matmulnbits-accuracy-level"
    if ops[:4] == ["QuantizeLinear", "DequantizeLinear", "QuantizeLinear", "DequantizeLinear"] and len(fused) < len(ops):
        return "RC1-double-qdq-remover"
    if ops and ops[0] == "QuantizeLinear" and any(a.get("block_size") for a in q_attrs):
        return "RC4-blocked-q-int32-overflow"
    if f["detail"].get("regime") == "normal" and any(o in fused for o in ("com.microsoft.QLinearAdd", "com.microsoft.QLinearMul")):
        return "RC5-qlinear-binary-overflow"
    if f["detail"].get("regime") == "extreme-scale" and f["detail"].get("example", {}).get("got") in (0, -128) \
            and f["detail"]["example"].get("want") in (255, 127) and any(o in fused for o in ("com.microsoft.QLinearAdd", "com.microsoft.QLinearMul")):
        return "RC5-qlinear-binary-overflow"
    if f["detail"].get("regime") == "extreme-scale":
        return "RC6-requant-float32-extreme-scales"
    return None


def main():
    camp = json.loads((ROOT / "results" / "campaign.json").read_text())
    out = {}
    for f in camp["findings"]:
        case = json.loads((ROOT / "findings" / f["id"] / "finding.json").read_text())["case"]
        rc = route(f, case)
        out[f["id"]] = dict(RC[rc], root_cause=rc) if rc else {"verdict": "untriaged", "root_cause": "untriaged"}
    (ROOT / "results" / "triage.json").write_text(json.dumps(out, indent=1))
    from collections import Counter
    print(Counter(v["root_cause"] for v in out.values()))


if __name__ == "__main__":
    main()
