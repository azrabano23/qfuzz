"""CLI smoke test: report rendering from the committed campaign results."""
import json
from pathlib import Path

from qfuzz.cli import main
from qfuzz.campaign import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_report_from_committed_results(tmp_path):
    out = tmp_path / "REPORT.md"
    assert main(["report", "--campaign", str(ROOT / "results" / "campaign.json"),
                 "--triage", str(ROOT / "results" / "triage.json"), "--out", str(out)]) == 0
    text = out.read_text()
    assert "# qfuzz campaign report" in text and "Minimized findings" in text


def test_summarize_and_show(capsys):
    recs = [{"seed": 1, "pattern": "p", "tags": [], "ms": 1,
             "verdict": {"cls": "mismatch", "levels": ["all"], "signature": "mismatch|all|X", "detail": {}, "stats": {}}},
            {"seed": 2, "pattern": "p", "tags": [], "ms": 1,
             "verdict": {"cls": "exact", "levels": [], "signature": "exact", "detail": {}, "stats": {"fusion_bitdiff": True}}}]
    s = summarize(recs, 1.0, 1, 0)
    assert s["counts"] == {"mismatch": 1, "exact": 1} and s["fusion_bitdiff_cases"] == 1
    assert s["signatures"][0]["seeds"] == [1]
    assert main(["show", "3"]) == 0
    assert '"cls"' in capsys.readouterr().out
    json.dumps(s)
