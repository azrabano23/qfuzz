"""Every committed findings/<id>/repro.py must run standalone.

Whether a discrepancy *reproduces* depends on the ORT version and on the CPU's
ISA (MLAS picks kernels at runtime), so a repro that runs cleanly but does not
reproduce is skipped rather than failed -- CI stays green on any x86/ARM runner
while still catching broken scripts.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

FINDINGS = sorted((Path(__file__).resolve().parents[1] / "findings").glob("*/repro.py"))


@pytest.mark.parametrize("script", FINDINGS, ids=[p.parent.name for p in FINDINGS])
def test_repro_script_runs(script):
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(report["levels"]) == {"disable", "extended", "all"}
    if not report["reproduced"]:
        pytest.skip(f"does not reproduce on onnxruntime {report['onnxruntime']} / this CPU")


def test_findings_exist():
    assert FINDINGS, "no committed findings/*/repro.py"
