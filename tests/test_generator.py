"""Generated models are valid ONNX and evaluable by the reference."""
import onnx
import pytest

from qfuzz import reference as ref
from qfuzz.case import Case
from qfuzz.generator import PATTERNS, generate


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_pattern_produces_valid_models(pattern):
    for seed in range(12):
        case = generate(seed, pattern)
        onnx.checker.check_model(case.to_model(), full_check=True)
        env = ref.evaluate(case)
        assert set(env) == set(case.outputs)


def test_generation_is_deterministic_and_json_roundtrips():
    a, b = generate(123), generate(123)
    assert a.dumps() == b.dumps()
    c = Case.from_json(a.to_json())
    assert c.dumps() == a.dumps()
    assert c.to_model().SerializeToString() == a.to_model().SerializeToString()


def test_generator_covers_edge_cases():
    tags = set()
    for seed in range(400):
        tags.update(generate(seed).meta["tags"])
    for t in ["ties", "scale-subnormal", "scale-huge", "zp-boundary", "blocked", "per-axis", "sat-boundary"]:
        assert t in tags, t
