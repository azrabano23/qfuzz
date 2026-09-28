"""The delta debugger shrinks a synthetic, injected bug to its essence."""
import numpy as np

import qfuzz.oracle as oracle
from qfuzz.generator import generate
from qfuzz.minimize import minimize
from qfuzz.runner import run_all as real_run_all


def buggy_run_all(case, *a, **k):
    """Pretend ORT at ENABLE_ALL mis-computes any QLinearMatMul whose A contains 77."""
    res = real_run_all(case, *a, **k)
    has_qlmm = any(n.op == "QLinearMatMul" for n in case.nodes)
    a_name = next((n.inputs[0] for n in case.nodes if n.op == "QLinearMatMul"), None)
    t = case.tensor(a_name) if a_name else None
    if has_qlmm and t is not None and (t.data == 77).any() and res["all"].outputs:
        o = case.outputs[0]
        out = res["all"].outputs[o].astype(np.int64)
        out = np.where(out > 100, out - 50, out + 50)  # far outside any rounding latitude
        res["all"].outputs[o] = out.astype(res["all"].outputs[o].dtype)
    return res


def test_minimizer_shrinks_injected_bug(monkeypatch):
    monkeypatch.setattr(oracle, "run_all", buggy_run_all)
    # find a seed whose generated QLinearMatMul case triggers the fake bug
    for seed in range(300):
        case = generate(seed, "qlinear_matmul")
        a = case.tensor(case.nodes[0].inputs[0])
        if a is not None and a.data.size > 20 and (a.data == 77).any():
            v = oracle.judge(case)
            if v.cls == "mismatch" and v.levels == ["all"] and v.detail["regime"] == "normal":
                break
    else:
        raise AssertionError("no seed triggered the synthetic bug")
    before = sum(t.data.size for t in case.inputs + case.inits)
    m, mv = minimize(case, v, budget_s=20)
    after = sum(t.data.size for t in m.inputs + m.inits)
    assert mv.cls == "mismatch" and mv.levels == ["all"]
    assert (m.tensor(m.nodes[0].inputs[0]).data == 77).any()
    assert after < before
    a_min = m.tensor(m.nodes[0].inputs[0]).data
    assert a_min.size <= 4, a_min.shape  # M and K shrunk to (almost) a single element
