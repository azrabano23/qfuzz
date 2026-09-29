"""Route the TVM campaign's non-clean signatures to hand-triaged groups.

Inputs:  results/tvm_campaign.json (qfuzz run --backend tvm)
         results/tvm_on_ort_findings.json (the 55 minimized ORT findings replayed on TVM, optional)
Outputs: results/tvm_triage.json and results/TVM_REPORT.md

The group texts were written after reading the TVM 0.27 Relax ONNX importer
(tvm/relax/frontend/onnx/onnx_frontend.py) and the failing cases. Every
tvm-error-build case was checked to contain a zero Q/DQ scale (88/88). A zero
scale is undefined in the ONNX spec, so that group is "spec-undefined input",
not a bug. Unrouted signatures stay "untriaged", and a non-empty untriaged list
fails the script.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qfuzz.oracle import FINDING_CLASSES  # noqa: E402
from qfuzz.report import render  # noqa: E402

GROUPS = {
    "T1-int4-initializer-import": dict(
        verdict="tvm-importer-bug",
        title="T1: any int4/uint4 initializer fails to import (TensorCopyFromBytes size mismatch)",
        summary="`_parse_graph_initializers` turns an INT4/UINT4 TensorProto into an ml_dtypes int4 array via "
                "onnx.numpy_helper.to_array (1 byte per element) and wraps it in relax.const, whose TVM int4 tensor "
                "is packed (2 elements per byte). The byte counts differ, and the import aborts with "
                "`Check failed: arr_size == nbytes`. It happens for int32_data and raw_data "
                "encodings alike, so every blocked int4 weight graph (the DQ->MatMul 'MatMulNBits' pattern) is "
                "untestable. It is a loud failure on a valid model, and no wrong numbers come out. No matching "
                "apache/tvm issue was found (searched 2026-09-29); related: #20124 (ONNX op coverage RFC).",
        match=r"^tvm-error-import\|.*TensorCopyFromBytes: size mismatch"),
    "T2-unsupported-feature": dict(
        verdict="unsupported",
        title="T2: features the importer rejects explicitly (not bugs)",
        summary="The importer raises a clear error for: blocked Q/DQ (`block_size>0`), float8 zero points in Q/DQ "
                "(E4M3FN/E5M2 are not in its allowed zero-point types), int4/uint4 as QuantizeLinear output, float8 "
                "`saturate=0`, and ops that have no converter (QLinearMatMul, QLinearConv, ConvInteger, "
                "com.microsoft.QLinearAdd/Mul). The TVM campaign avoids the last group by generating only patterns "
                "made of supported ops. The replay of ORT findings still hits it.",
        match=r"^tvm-unsupported\|"),
    "T3-zero-scale-compile-error": dict(
        verdict="spec-undefined",
        title="T3: y_scale = 0 fails at compile time (Divide by zero in constant folding)",
        summary="All 88 cases have a zero Q/DQ scale. The ONNX spec's formula divides by it, and qfuzz treats a zero "
                "scale as undefined (see README 'Spec ambiguities'). ORT accepts these models and returns some value "
                "(0 or -128 in the cases checked), and TVM refuses to compile them. Both are acceptable, so this is not a bug.",
        match=r"^tvm-error-build\|.*Divide by zero"),
}


def main() -> int:
    camp = json.loads((ROOT / "results" / "tvm_campaign.json").read_text())
    groups = {k: dict(v, hits=0, signatures=[]) for k, v in GROUPS.items()}
    untriaged, findings = [], []
    for s in camp["signatures"]:
        if s["cls"] in FINDING_CLASSES:
            findings.append(s)
            continue
        for k, g in GROUPS.items():
            if re.search(g["match"], s["signature"]):
                groups[k]["hits"] += s["count"]
                groups[k]["signatures"].append({"signature": s["signature"], "count": s["count"], "seeds": s["seeds"]})
                break
        else:
            untriaged.append(s["signature"])
    replay = ROOT / "results" / "tvm_on_ort_findings.json"
    out = {"tvm": camp["env"].get("tvm"), "cases": camp["config"]["cases"], "counts": camp["counts"],
           "tvm_vs_ort_disable": camp.get("tvm_vs_ort_disable"), "findings": findings,
           "groups": {k: {kk: vv for kk, vv in g.items() if kk != "match"} for k, g in groups.items()},
           "untriaged": untriaged,
           "ort_findings_replayed_on_tvm": json.loads(replay.read_text())["by_root_cause"] if replay.exists() else None}
    (ROOT / "results" / "tvm_triage.json").write_text(json.dumps(out, indent=1))

    md = render(camp, {})
    md = md.split("## Minimized findings")[0]
    L = ["## TVM triage\n",
         f"Findings (mismatch / off-by-one / nan-inf / crash): **{len(findings)}**.\n",
         "| group | verdict | cases |\n|---|---|---:|"]
    for k, g in groups.items():
        L.append(f"| {g['title']} | {g['verdict']} | {g['hits']} |")
    L.append("")
    for k, g in groups.items():
        L.append(f"### {g['title']}\n\n{g['summary']}\n")
    if out["ort_findings_replayed_on_tvm"]:
        L.append("## ORT's 55 minimized findings replayed on TVM\n")
        L.append("| ORT root cause | TVM verdicts |\n|---|---|")
        for rc, c in sorted(out["ort_findings_replayed_on_tvm"].items()):
            L.append(f"| {rc} | {', '.join(f'{k}: {v}' for k, v in sorted(c.items()))} |")
        L.append("")
    (ROOT / "results" / "TVM_REPORT.md").write_text(md + "\n".join(L) + "\n")
    print(json.dumps({k: g["hits"] for k, g in groups.items()}), "findings:", len(findings), "untriaged:", untriaged)
    return 1 if untriaged else 0


if __name__ == "__main__":
    sys.exit(main())
