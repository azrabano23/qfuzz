"""The paste-ready upstream reports in upstream/ must stay runnable and in sync.

Each upstream/*.py script builds its model inline, prints expected vs actual and
ends with ``assert <cond>, "<message>"``. While the bug is present, it exits with
``AssertionError: <message>``. Same policy as test_repros.py: a script that runs
cleanly (the bug does not reproduce on this ORT build or CPU) is skipped, not failed.
A bare ``AssertionError`` comes from a sanity check on the expected values, and it
fails the test, as does any other error.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"
SCRIPTS = sorted(UPSTREAM.glob("*.py"))
ALLOWED_IMPORTS = {"numpy", "onnx", "onnxruntime"}


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_upstream_script_reproduces(script):
    proc = subprocess.run([sys.executable, script.name], cwd=UPSTREAM, capture_output=True, text=True, timeout=120)
    if proc.returncode == 0:
        pytest.skip(f"{script.name} does not reproduce here:\n{proc.stdout[-1000:]}")
    last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    assert last.startswith("AssertionError: "), f"script failed for another reason:\n{proc.stderr[-2000:]}"
    assert "expected" in proc.stdout


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_upstream_script_is_short_and_self_contained(script):
    src = script.read_text()
    assert len(src.splitlines()) <= 40
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module.split(".")[0])
    assert mods <= ALLOWED_IMPORTS, mods


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_upstream_report_embeds_current_script(script):
    report = script.with_suffix(".md")
    assert report.exists(), f"missing {report.name}"
    assert script.read_text().rstrip() in report.read_text(), f"{report.name} embeds a stale copy of {script.name}"


def test_upstream_readme_lists_every_report():
    readme = (UPSTREAM / "README.md").read_text()
    for script in SCRIPTS:
        assert script.with_suffix(".md").name in readme
