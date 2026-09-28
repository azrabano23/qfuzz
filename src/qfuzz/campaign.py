"""Fuzzing campaign: crash-isolated parallel workers, dedupe, minimization."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .case import Case
from .generator import GENERATOR_VERSION, generate
from .oracle import FINDING_CLASSES, SEVERITY, judge


# ------------------------------------------------------------------ worker
def worker_main(argv: list[str]) -> int:
    """python -m qfuzz.campaign worker START STOP OUT DEADLINE"""
    start, stop, out, deadline = int(argv[0]), int(argv[1]), argv[2], float(argv[3])
    with open(out, "a", buffering=1) as f:
        for seed in range(start, stop):
            if time.time() > deadline:
                break
            f.write(json.dumps({"begin": seed}) + "\n")
            f.flush()
            t0 = time.time()
            try:
                case = generate(seed)
                v = judge(case)
                rec = {"seed": seed, "pattern": case.meta["pattern"], "tags": case.meta.get("tags", []),
                       "verdict": v.to_json(), "ms": round(1000 * (time.time() - t0), 1)}
            except Exception as e:  # noqa: BLE001 - a qfuzz bug, not an ORT bug
                rec = {"seed": seed, "pattern": "?", "tags": [], "ms": 0,
                       "verdict": {"cls": "qfuzz-error", "levels": [], "signature": f"qfuzz-error|{type(e).__name__}",
                                   "detail": {"error": repr(e)[:300]}, "stats": {}}}
            f.write(json.dumps(rec) + "\n")
    return 0


def _run_shard(start: int, stop: int, out: Path, deadline: float) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "qfuzz.campaign", "worker", str(start), str(stop), str(out),
                             str(deadline)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_shard(path: Path) -> tuple[list[dict], int | None]:
    """Returns (records, seed_in_flight_or_None)."""
    recs, pending = [], None
    if not path.exists():
        return recs, None
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "begin" in d:
            pending = d["begin"]
        else:
            recs.append(d)
            pending = None
    return recs, pending


def run_campaign(n: int, workers: int, seed_start: int, time_limit: float, work_dir: Path, log=print) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    for p in work_dir.glob("shard*.jsonl"):
        p.unlink()
    deadline = time.time() + time_limit
    t0 = time.time()
    per = (n + workers - 1) // workers
    shards = []
    for w in range(workers):
        lo, hi = seed_start + w * per, min(seed_start + (w + 1) * per, seed_start + n)
        if lo < hi:
            shards.append([lo, hi, work_dir / f"shard{w}.jsonl", None])
    crashes: list[dict] = []
    for s in shards:
        s[3] = _run_shard(s[0], s[1], s[2], deadline)
    while any(s[3] is not None for s in shards):
        time.sleep(0.5)
        for s in shards:
            p = s[3]
            if p is None or p.poll() is None:
                continue
            recs, pending = _read_shard(s[2])
            if p.returncode != 0 and pending is not None:
                # the process died inside a case: record it as a crash and continue after it
                crashes.append({"seed": pending, "returncode": p.returncode})
                log(f"worker crashed on seed {pending} (rc={p.returncode}); restarting")
                with open(s[2], "a") as f:
                    f.write(json.dumps({"seed": pending, "pattern": "?", "tags": [], "ms": 0, "verdict": {
                        "cls": "crash", "levels": [], "signature": f"crash|rc={p.returncode}",
                        "detail": {"returncode": p.returncode}, "stats": {}}}) + "\n")
                if pending + 1 < s[1] and time.time() < deadline:
                    s[3] = _run_shard(pending + 1, s[1], s[2], deadline)
                    continue
            s[3] = None
        done = sum(len(_read_shard(s[2])[0]) for s in shards)
        if int(time.time() - t0) % 30 == 0:
            log(f"[{time.time() - t0:5.0f}s] {done}/{n} cases")
    records = []
    for s in shards:
        records += _read_shard(s[2])[0]
    return summarize(records, time.time() - t0, workers, seed_start)


def env_info() -> dict:
    import onnx
    import onnxruntime as ort
    flags = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = line
                break
    except OSError:
        pass
    isa = [f for f in ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "amx_tile") if f" {f}" in flags]
    model = ""
    try:
        model = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                     if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "machine": platform.machine(), "cpu": model, "isa": isa,
            "generator_version": GENERATOR_VERSION}


def summarize(records: list[dict], elapsed: float, workers: int, seed_start: int) -> dict:
    by_cls = collections.Counter(r["verdict"]["cls"] for r in records)
    by_pattern: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    sigs: dict[str, dict] = {}
    fusion_bitdiff = 0
    for r in records:
        v = r["verdict"]
        by_pattern[r["pattern"]][v["cls"]] += 1
        if v.get("stats", {}).get("fusion_bitdiff"):
            fusion_bitdiff += 1
        if v["cls"] in FINDING_CLASSES or v["cls"] in ("qfuzz-error", "invalid-model", "ref-unsupported"):
            s = sigs.setdefault(v["signature"], {"signature": v["signature"], "cls": v["cls"], "count": 0,
                                                 "seeds": [], "example": v.get("detail")})
            s["count"] += 1
            if len(s["seeds"]) < 5:
                s["seeds"].append(r["seed"])
    order = {c: i for i, c in enumerate(SEVERITY + ["qfuzz-error"])}
    sig_list = sorted(sigs.values(), key=lambda s: (order.get(s["cls"], 99), -s["count"]))
    return {
        "env": env_info(),
        "config": {"cases": len(records), "workers": workers, "seed_start": seed_start,
                   "elapsed_s": round(elapsed, 1), "levels": ["disable", "extended", "all"]},
        "counts": dict(by_cls),
        "fusion_bitdiff_cases": fusion_bitdiff,
        "by_pattern": {k: dict(v) for k, v in sorted(by_pattern.items())},
        "signatures": sig_list,
        "findings": [],
    }


# ---------------------------------------------------------- minimization
def finding_id(cls: str, ops: list[str], sig: str) -> str:
    slug = {"mismatch": "mm", "off-by-one": "obo", "nan-inf": "nan", "rejects-valid-model": "rej",
            "error-level-dependent": "lvl", "crash": "crash"}.get(cls, cls)
    core = [o for o in ops if o not in ("QuantizeLinear", "DequantizeLinear")] or ops
    h = hashlib.sha1(sig.encode()).hexdigest()[:6]
    return f"{slug}-{'-'.join(core)[:40]}-{h}".lower()


def minimize_campaign(campaign: dict, findings_dir: Path, per_sig: int = 1, budget_s: float = 60.0,
                      log=print) -> list[dict]:
    from .minimize import minimize
    from .repro import describe, write_repro
    from .runner import run_all

    findings: dict[str, dict] = {}
    for s in campaign["signatures"]:
        if s["cls"] not in FINDING_CLASSES or s["cls"] == "crash":
            continue
        for seed in s["seeds"][:per_sig]:
            case = generate(seed)
            v0 = judge(case)
            if v0.signature != s["signature"]:
                log(f"seed {seed}: not reproducible in-process ({v0.signature})")
                continue
            t0 = time.time()
            m, mv = minimize(case, v0, budget_s=budget_s)
            ops = [n.op for n in m.nodes]
            msig = f"{mv.cls}|{','.join(mv.levels)}|{'>'.join(ops)}|{mv.signature.split('|', 3)[-1]}"
            log(f"seed {seed}: {s['signature']} -> {len(m.nodes)} nodes, {sum(t.data.size for t in m.inputs + m.inits)} "
                f"values in {time.time() - t0:.1f}s")
            if msig in findings:
                findings[msig]["raw_signatures"].append(s["signature"])
                findings[msig]["hits"] += s["count"]
                continue
            fid = finding_id(mv.cls, ops, msig)
            fused = {lv: r.fused_ops for lv, r in run_all(m, record_fused=True).items()}
            findings[msig] = {"id": fid, "cls": mv.cls, "levels": mv.levels, "min_signature": msig,
                              "raw_signatures": [s["signature"]], "hits": s["count"], "seed": seed,
                              "ops": ops, "fused_ops": fused, "detail": mv.detail, "case": m.to_json(),
                              "description": describe(m)}
    out = []
    for f in findings.values():
        d = findings_dir / f["id"]
        d.mkdir(parents=True, exist_ok=True)
        case = Case.from_json(f["case"])
        title = f"{f['cls']} in {' -> '.join(f['ops'])} (levels: {', '.join(f['levels'])})"
        (d / "repro.py").write_text(write_repro(case, judge(case), f["id"], title))
        (d / "finding.json").write_text(json.dumps(f, indent=1, allow_nan=True))
        out.append({k: f[k] for k in ("id", "cls", "levels", "min_signature", "raw_signatures", "hits", "seed",
                                       "ops", "fused_ops", "detail", "description")})
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        sys.exit(worker_main(sys.argv[2:]))
