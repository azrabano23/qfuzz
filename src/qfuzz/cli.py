"""qfuzz command line: run / minimize / report / show."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # qfuzz/ project dir (src layout, editable install)
if not (ROOT / "pyproject.toml").exists():
    ROOT = Path.cwd()


def cmd_run(a) -> int:
    from .campaign import run_campaign
    res = run_campaign(a.cases, a.workers, a.seed_start, a.time_limit, Path(a.work),
                       log=lambda m: print(m, file=sys.stderr), backend=a.backend)
    out = Path(a.out)
    if out.exists():  # keep previously minimized findings if the campaign is re-run
        old = json.loads(out.read_text())
        res["findings"] = old.get("findings", []) if a.keep_findings else []
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=True))
    print(json.dumps({"cases": res["config"]["cases"], "counts": res["counts"],
                      "signatures": len(res["signatures"])}, indent=1))
    return 0


def cmd_minimize(a) -> int:
    from .campaign import minimize_campaign
    path = Path(a.campaign)
    camp = json.loads(path.read_text())
    fs = minimize_campaign(camp, Path(a.findings), per_sig=a.per_sig, budget_s=a.budget,
                           log=lambda m: print(m, file=sys.stderr))
    camp["findings"] = fs
    path.write_text(json.dumps(camp, indent=1, allow_nan=True))
    print(f"{len(fs)} minimized findings written to {a.findings}")
    return 0


def cmd_report(a) -> int:
    from .report import write_report
    write_report(Path(a.campaign), Path(a.triage) if a.triage else None, Path(a.out))
    print(f"wrote {a.out}")
    return 0


def cmd_show(a) -> int:
    from .generator import generate
    from .oracle import judge
    from .repro import describe
    case = generate(a.seed, a.pattern)
    if a.backend == "tvm":
        from . import tvm_runner
        v = tvm_runner.judge(case)
    else:
        v = judge(case)
    print(describe(case))
    print(json.dumps(v.to_json(), indent=1, allow_nan=True))
    if a.minimize:
        from .minimize import minimize
        m, mv = minimize(case, v, budget_s=a.budget)
        print("--- minimized ---")
        print(describe(m))
        print(json.dumps(mv.to_json(), indent=1, allow_nan=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qfuzz", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a fuzzing campaign")
    r.add_argument("--cases", type=int, default=5000)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--seed-start", type=int, default=0)
    r.add_argument("--time-limit", type=float, default=900.0, help="seconds")
    r.add_argument("--work", default=str(ROOT / "work"))
    r.add_argument("--out", default=str(ROOT / "results" / "campaign.json"))
    r.add_argument("--keep-findings", action="store_true")
    r.add_argument("--backend", choices=["ort", "tvm"], default="ort",
                   help="ort: 3 optimization levels (default); tvm: Apache TVM Relax, llvm target (optional dep)")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("minimize", help="minimize one case per signature and write findings/<id>/repro.py")
    m.add_argument("--campaign", default=str(ROOT / "results" / "campaign.json"))
    m.add_argument("--findings", default=str(ROOT / "findings"))
    m.add_argument("--per-sig", type=int, default=1)
    m.add_argument("--budget", type=float, default=60.0, help="seconds per case")
    m.set_defaults(fn=cmd_minimize)
    rp = sub.add_parser("report", help="render results/REPORT.md")
    rp.add_argument("--campaign", default=str(ROOT / "results" / "campaign.json"))
    rp.add_argument("--triage", default=str(ROOT / "results" / "triage.json"))
    rp.add_argument("--out", default=str(ROOT / "results" / "REPORT.md"))
    rp.set_defaults(fn=cmd_report)
    s = sub.add_parser("show", help="generate, judge (and optionally minimize) one seed")
    s.add_argument("seed", type=int)
    s.add_argument("--pattern")
    s.add_argument("--minimize", action="store_true")
    s.add_argument("--backend", choices=["ort", "tvm"], default="ort")
    s.add_argument("--budget", type=float, default=30.0)
    s.set_defaults(fn=cmd_show)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
