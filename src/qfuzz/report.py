"""Render results/campaign.json (+ curated triage) into a Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

from .oracle import FINDING_CLASSES, SEVERITY

VERDICT_TEXT = {
    "ort-bug": "Real ORT bug",
    "ort-bug-degenerate": "Real ORT bug (only with degenerate/extreme inputs)",
    "by-design": "Intentional ORT behaviour (documented trade-off), not spec-conformant",
    "spec-ambiguity": "Spec ambiguity (both readings defensible)",
    "unsupported": "Valid model rejected (unsupported type/feature)",
    "reference-bug": "Bug in qfuzz's reference (fixed)",
    "untriaged": "Not yet triaged",
}


def load_triage(path: Path | None) -> dict:
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def render(campaign: dict, triage: dict) -> str:
    env, cfg, counts = campaign["env"], campaign["config"], campaign["counts"]
    total = cfg["cases"]
    L = []
    L.append("# qfuzz campaign report\n")
    L.append(f"ONNX Runtime **{env['onnxruntime']}** (CPU EP), onnx {env['onnx']}, numpy {env['numpy']}, "
             f"Python {env['python']} on `{env['cpu']}` (ISA: {', '.join(env['isa'])}).\n")
    if env.get("backend") == "tvm":
        L.append(f"Backend: **Apache TVM {env.get('tvm')}** (Relax ONNX frontend, target `{env.get('tvm_target')}`).\n")
    L.append(f"**{total}** generated cases, {cfg['workers']} workers, {cfg['elapsed_s']} s wall clock, "
             f"each executed at {', '.join(cfg['levels'])} and checked against the exact "
             f"reference.\n")
    L.append("## Verdicts\n")
    L.append("| class | cases | share |\n|---|---:|---:|")
    for c in SEVERITY + ["qfuzz-error"] + sorted(k for k in counts if k not in SEVERITY + ["qfuzz-error"]):
        if c in counts:
            L.append(f"| {'**' + c + '**' if c in FINDING_CLASSES else c} | {counts[c]} | {100 * counts[c] / total:.2f}% |")
    if "tvm_vs_ort_disable" in campaign:
        t = campaign["tvm_vs_ort_disable"]
        L.append(f"\nTVM output bit-identical to ORT `ORT_DISABLE_ALL`: **{t['bit_identical']}** cases; differs: "
                 f"**{t['differ']}**; not compared (TVM or ORT error): {t['not_compared']}.\n")
    else:
        L.append(f"\nCases where fusion changed at least one output bit (oracle a, any level vs disable): "
                 f"**{campaign['fusion_bitdiff_cases']}** ({100 * campaign['fusion_bitdiff_cases'] / total:.2f}%).\n")
    L.append("## Per pattern\n")
    allc = SEVERITY + sorted({c for v in campaign["by_pattern"].values() for c in v} - set(SEVERITY))
    cols = [c for c in allc if any(c in v for v in campaign["by_pattern"].values())]
    L.append("| pattern | " + " | ".join(cols) + " |")
    L.append("|---|" + "---:|" * len(cols))
    for p, v in campaign["by_pattern"].items():
        L.append(f"| {p} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |")
    L.append("")
    sigs = [s for s in campaign["signatures"] if s["cls"] in FINDING_CLASSES or s["cls"].startswith("tvm-")]
    L.append(f"## Raw signatures ({len(sigs)} distinct, before minimization)\n")
    L.append("| class | hits | signature | example seeds |\n|---|---:|---|---|")
    for s in sigs:
        sig = s['signature'][:110].replace('|', '\\|')
        L.append(f"| {s['cls']} | {s['count']} | `{sig}` | {', '.join(map(str, s['seeds'][:3]))} |")
    L.append("")
    fs = campaign.get("findings", [])
    L.append(f"## Minimized findings ({len(fs)})\n")
    L.append("Deduplicated by the signature of the *minimized* graph. Triage verdicts are curated by hand in "
             "`results/triage.json` after reading the ONNX spec and the minimized repro.\n")
    groups: dict[str, list] = {}
    fs = sorted(fs, key=lambda f: (triage.get(f["id"], {}).get("root_cause", "~"), -f["hits"]))
    for f in fs:
        t = triage.get(f["id"], {})
        groups.setdefault(t.get("root_cause", f["id"]), []).append((f, t))
    L.append("### Root-cause summary\n")
    L.append("| root cause | verdict | minimized findings | raw hits (cases) | scale regimes |\n|---|---|---:|---:|---|")
    for rc, items in groups.items():
        t0 = items[0][1]
        regimes = sorted({f["detail"].get("regime", "-") for f, _ in items})
        L.append(f"| {t0.get('title', rc)[:95]} | {VERDICT_TEXT.get(t0.get('verdict', 'untriaged'))} | "
                 f"{len(items)} | {sum(f['hits'] for f, _ in items)} | {', '.join(regimes)} |")
    L.append("")
    L.append("| finding | class | levels | graph | hits | triage | root cause |\n|---|---|---|---|---:|---|---|")
    for rc, items in groups.items():
        for f, t in items:
            L.append(f"| [`{f['id']}`](../findings/{f['id']}/repro.py) | {f['cls']} | {','.join(f['levels'])} | "
                     f"{' → '.join(f['ops'])} | {f['hits']} | {VERDICT_TEXT.get(t.get('verdict', 'untriaged'))} | {rc} |")
    L.append("")
    for rc, items in groups.items():
        t0 = items[0][1]
        L.append(f"### {t0.get('title', rc)}\n")
        if t0:
            L.append(f"**Verdict:** {VERDICT_TEXT.get(t0.get('verdict', 'untriaged'))}.  ")
            if t0.get("known"):
                L.append(f"**Known upstream?** {t0['known']}  ")
            L.append("")
            if t0.get("summary"):
                L.append(t0["summary"] + "\n")
        for f, _ in items:
            ex = f["detail"].get("example") or {}
            L.append(f"- `{f['id']}` ({f['hits']} raw hits; fused at extended: "
                     f"`{' '.join(f['fused_ops'].get('extended', []))}`)")
            if ex:
                L.append(f"  - example: index {ex.get('index')} got **{ex.get('got')}**, exact **{ex.get('want')}**, "
                         f"allowed slack {ex.get('slack')}")
            if f["detail"].get("errors"):
                L.append(f"  - error: `{next(iter(f['detail']['errors'].values()))[:200]}`")
            L.append("  ```\n  " + f["description"].replace("\n", "\n  ") + "\n  ```")
        L.append("")
    return "\n".join(L) + "\n"


def write_report(campaign_path: Path, triage_path: Path | None, out: Path) -> str:
    campaign = json.loads(campaign_path.read_text())
    text = render(campaign, load_triage(triage_path))
    out.write_text(text)
    return text
