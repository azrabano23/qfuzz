"""Differential oracles and classification.

Three oracles are applied to every case:

(a) *fusion oracle* -- ORT with graph optimizations disabled vs. ORT at
    ``ENABLE_EXTENDED`` / ``ENABLE_ALL`` (where QDQ fusions happen).  Bit
    differences are recorded as statistics; they are only *findings* when one
    of the levels also violates (b)/(c), or when a level errors while another
    does not.
(b) *reference oracle* -- every level vs. the exact reference interpreter.
(c) *bound oracle* -- a disagreement is acceptable iff it is within the
    reference's propagated slack (see :mod:`qfuzz.reference`).  Far from a
    rounding tie the slack is 0: integer outputs must then match exactly.

Classes, most severe first (``FINDING_CLASSES`` are reported):
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import onnx

from . import reference as ref
from .case import Case, FLOAT8_TYPES
from .runner import OrtResult, run_all

SEVERITY = [
    "crash",               # process died (segfault/abort) inside ORT
    "error-level-dependent",  # some opt levels fail, others succeed
    "nan-inf",             # NaN/inf where the reference is finite (or vice versa)
    "mismatch",            # beyond bound by >= 2 LSB (int) or beyond float bound
    "off-by-one",          # exactly 1 LSB beyond bound (reference says: no tie nearby)
    "rejects-valid-model", # all levels refuse a model that onnx.checker accepts
    "within-bound",        # differs from the ideal value, but within spec latitude
    "exact",               # bit-exact at every level
    "ref-unsupported",     # reference can't evaluate; only oracle (a) applied
    "invalid-model",       # generator produced a model the checker rejects
]
FINDING_CLASSES = {"crash", "error-level-dependent", "nan-inf", "mismatch", "off-by-one", "rejects-valid-model"}
__doc__ += "\n".join(f"  * ``{c}``" for c in SEVERITY)


def _rank(c: str) -> int:
    return SEVERITY.index(c)


@dataclass
class OutputCmp:
    cls: str
    max_excess: float = 0.0  # max(|got-ref| - slack)
    max_diff: float = 0.0
    n_bad: int = 0
    n_inexact: int = 0
    example: dict | None = None


@dataclass
class Verdict:
    cls: str
    levels: list[str]
    output: str | None
    producer: str | None
    signature: str
    detail: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"cls": self.cls, "levels": self.levels, "output": self.output,
                "producer": self.producer, "signature": self.signature,
                "detail": self.detail, "stats": self.stats}

    @property
    def is_finding(self) -> bool:
        return self.cls in FINDING_CLASSES


def compare(got: np.ndarray, rv: ref.Val) -> OutputCmp:
    want = rv.value
    slack = np.broadcast_to(rv.slack, np.shape(want))
    got = np.asarray(got)
    if got.shape != np.shape(want):
        return OutputCmp("mismatch", np.inf, np.inf, int(np.size(want)),
                         example={"reason": f"shape {got.shape} != {np.shape(want)}"})
    if rv.dtype in FLOAT8_TYPES:
        got = ref.f8_bits_to_f64(got.view(np.uint8), rv.dtype)
    if rv.is_int:
        g = got.astype(np.int64)
        diff = np.abs(g - want).astype(np.float64)
        nan_bad = np.zeros(diff.shape, bool)
    else:
        # +/-inf is compared as +/-2**128 (the first value float32 rounds to inf), so a
        # finite result within slack of an overflowing ideal value is accepted and
        # vice versa; NaN must match NaN.
        g = got.astype(np.float64)
        w = want.astype(np.float64)
        gc = np.where(np.isinf(g), np.sign(g) * 2.0**128, g)
        wc = np.where(np.isinf(w), np.sign(w) * 2.0**128, w)
        gnan, wnan = np.isnan(g), np.isnan(w)
        same_special = (gnan & wnan) | (np.isinf(g) & np.isinf(w) & (np.sign(g) == np.sign(w)))
        with np.errstate(invalid="ignore"):
            diff = np.where(same_special | gnan | wnan, 0.0, np.abs(gc - wc))
        nan_bad = (gnan ^ wnan) & ~np.isinf(slack)
        nan_bad |= (np.isinf(g) != np.isinf(w)) & ~same_special & (diff > slack) & ~np.isinf(slack)
    excess = np.where(np.isinf(slack), 0.0, diff - slack)
    bad = (excess > 0) | nan_bad
    n_inexact = int(np.count_nonzero(diff > 0))
    if nan_bad.any():
        idx = tuple(int(i) for i in np.argwhere(nan_bad)[0])
        return OutputCmp("nan-inf", np.inf, float(np.max(diff)), int(bad.sum()), n_inexact,
                         _example(idx, g, want, slack))
    if not bad.any():
        return OutputCmp("exact" if n_inexact == 0 else "within-bound", 0.0, float(diff.max(initial=0)), 0, n_inexact)
    idx = tuple(int(i) for i in np.unravel_index(np.argmax(np.where(bad, excess, -1)), excess.shape))
    mx = float(excess[idx])
    if rv.is_int:
        cls = "off-by-one" if float(np.max(np.where(bad, diff, 0))) <= 1 else "mismatch"
    else:
        cls = "mismatch"
    return OutputCmp(cls, mx, float(diff.max()), int(bad.sum()), n_inexact, _example(idx, g, want, slack))


def _num(x):
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return str(x)
    return x if not x.is_integer() else int(x)


def _example(idx, g, want, slack) -> dict:
    return {"index": list(idx), "got": _num(g[idx]), "want": _num(want[idx]), "slack": _num(slack[idx])}


def _norm_err(msg: str) -> str:
    m = re.sub(r"\d+", "N", msg)
    m = re.sub(r"Name:'[^']*'", "", m)
    return m[:120]


def check_model(case: Case) -> str | None:
    try:
        onnx.checker.check_model(case.to_model(), full_check=True)
    except Exception as e:  # noqa: BLE001
        return str(e)[:300]
    return None


def judge(case: Case, results: dict[str, OrtResult] | None = None) -> Verdict:
    pattern = case.meta.get("pattern", "?")
    ops = "+".join(case.op_types())
    if results is None:
        results = run_all(case)
    levels = list(results)
    errs = {lv: r for lv, r in results.items() if r.error}
    stats: dict = {"fusion_bitdiff": False, "inexact_levels": []}

    if errs:
        chk = check_model(case)
        if chk:
            return Verdict("invalid-model", list(errs), None, None, f"invalid-model|{pattern}",
                           {"checker": chk, "ort": next(iter(errs.values())).error})
        e0 = next(iter(errs.values()))
        if len(errs) == len(results):
            cls = "rejects-valid-model"
        else:
            cls = "error-level-dependent"
        sig = f"{cls}|{','.join(sorted(errs))}|{ops}|{_norm_err(e0.error)}"
        return Verdict(cls, sorted(errs), None, None, sig,
                       {"errors": {lv: r.error for lv, r in errs.items()}, "stage": e0.stage})

    # (a) fusion oracle: bitwise comparison between levels
    base = results["disable"].outputs if "disable" in results else None
    if base is not None:
        for lv, r in results.items():
            for o in case.outputs:
                a, b = np.asarray(base[o]), np.asarray(r.outputs[o])
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                    stats["fusion_bitdiff"] = True
                    stats.setdefault("fusion_bitdiff_levels", []).append(lv)
    try:
        want = ref.evaluate(case)
    except ref.RefError as e:
        cls = "ref-unsupported"
        return Verdict(cls, [], None, None, f"{cls}|{ops}", {"ref_error": str(e)}, stats)

    worst: tuple[int, str, str, OutputCmp] | None = None
    bad_levels: dict[str, set] = {}
    for lv, r in results.items():
        for o in case.outputs:
            c = compare(r.outputs[o], want[o])
            if c.cls == "within-bound":
                stats["inexact_levels"].append(lv)
            if c.cls in FINDING_CLASSES:
                bad_levels.setdefault(c.cls, set()).add(lv)
            if worst is None or _rank(c.cls) < _rank(worst[3].cls) or (
                    c.cls == worst[3].cls and c.max_excess > worst[3].max_excess):
                worst = (0, lv, o, c)
    assert worst is not None
    _, lv, o, c = worst
    prod = case.producer(o)
    producer = prod.op if prod else None
    out_dtype = want[o].dtype
    lvls = sorted(bad_levels.get(c.cls, {lv}))
    stats["inexact_levels"] = sorted(set(stats["inexact_levels"]))
    regime = scale_regime(case)
    sig = f"{c.cls}|{','.join(lvls)}|{_root_op(case, o)}|{out_dtype}|{regime}|{pattern}"
    detail = {"regime": regime, "output": o, "level": lv, "max_excess": c.max_excess, "max_diff": c.max_diff,
              "n_bad": c.n_bad, "n_inexact": c.n_inexact, "example": c.example}
    return Verdict(c.cls, lvls, o, producer, sig, detail, stats)


SCALE_SLOTS = {"QuantizeLinear": (1,), "DequantizeLinear": (1,), "QLinearMatMul": (1, 4, 6),
               "QLinearConv": (1, 4, 6), "QLinearAdd": (1, 4, 6), "QLinearMul": (1, 4, 6)}
EXTREME_LO, EXTREME_HI = 1e-12, 1e12


def scale_regime(case: Case) -> str:
    """'normal' unless some quantization scale is extreme (outside [1e-12, 1e12], or 0/inf).

    Findings that need such scales are real disagreements, but of much lower
    practical relevance; keeping the regime in the signature stops the
    minimizer from trading a realistic bug for a degenerate one.
    """
    names = set()
    for n in case.nodes:
        for i in SCALE_SLOTS.get(n.op, ()):
            if i < len(n.inputs) and n.inputs[i]:
                names.add(n.inputs[i])
    for t in case.inputs + case.inits:
        if t.name in names and t.dtype == "float32":
            a = np.abs(t.data.astype(np.float64))
            if np.any((a < EXTREME_LO) | (a > EXTREME_HI) | ~np.isfinite(a)):
                return "extreme-scale"
    return "normal"


def _root_op(case: Case, out: str) -> str:
    """The first non-(Q/DQ/observer) op upstream of ``out`` -- the 'compute' op."""
    seen = []
    frontier = [out]
    while frontier:
        name = frontier.pop()
        n = case.producer(name)
        if n is None:
            continue
        seen.append(n.op)
        frontier.extend(i for i in n.inputs if i)
    compute = [s for s in seen if s not in ("QuantizeLinear", "DequantizeLinear", "Identity")]
    return compute[0] if compute else ("+".join(sorted(set(seen))) or "?")
