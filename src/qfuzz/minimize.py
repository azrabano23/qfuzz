"""Delta-debugging minimizer for failing Cases.

The predicate is "the case still produces a verdict with the same class, the
same set of failing optimization levels and the same root op (and, for error
classes, the same normalized error message)".  Reductions, applied to a
fixpoint, cheapest-first:

1. **cut**: replace an intermediate tensor by a constant/graph input holding
   its reference value, then drop dead upstream nodes;
2. **trim outputs**: expose an earlier tensor as the output (drops the tail);
3. **shrink dims**: slice every tensor axis that carries a given dimension
   label (keeps coupled dims such as MatMul's K consistent);
4. **simplify values**: ddmin over tensor elements, replacing chunks by a
   "boring" value (0 / zero-point), then rounding floats to few significant
   digits so the repro is readable.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import reference as ref
from .case import DTYPES, FLOAT8_TYPES, OPAQUE_TYPES, Case, TensorSpec
from .oracle import Verdict, _norm_err, _root_op, judge


def same_bug(target: Verdict) -> Callable[[Case], bool]:
    def pred(c: Case) -> bool:
        try:
            v = judge(c)
        except Exception:  # noqa: BLE001 - a crashing reduction is not interesting
            return False
        if v.cls != target.cls or set(v.levels) != set(target.levels):
            return False
        if v.cls in ("rejects-valid-model", "error-level-dependent"):
            e1 = next(iter(v.detail.get("errors", {}).values()), "")
            e0 = next(iter(target.detail.get("errors", {}).values()), "")
            return _norm_err(e1) == _norm_err(e0)
        if target.output and v.output:
            t_parts, v_parts = target.signature.split("|"), v.signature.split("|")
            return t_parts[2:5] == v_parts[2:5]  # root op, output dtype, scale regime
        return True
    return pred


class Minimizer:
    def __init__(self, case: Case, pred: Callable[[Case], bool], budget_s: float = 120.0, log=None):
        self.best = case.prune()
        self.pred = pred
        self.deadline = time.time() + budget_s
        self.tests = 0
        self.log = log or (lambda *a: None)

    def _try(self, c: Case) -> bool:
        if time.time() > self.deadline:
            return False
        try:
            c = c.prune()
            if c.size() > self.best.size() and len(c.nodes) >= len(self.best.nodes):
                return False
        except Exception:  # noqa: BLE001
            return False
        self.tests += 1
        if self.pred(c):
            self.best = c
            return True
        return False

    # ---------------------------------------------------------------- passes
    def pass_cut(self) -> bool:
        progress = False
        try:
            env = ref.evaluate(self.best, keep_all=True)
        except Exception:  # noqa: BLE001
            return False
        produced = [o for n in self.best.nodes for o in n.outputs if o and o not in self.best.outputs]
        for name in produced:
            if name not in env:
                continue
            v = env[name]
            if not np.all(np.isfinite(v.value)) and v.dtype == "float32":
                pass  # still allowed: inf values are legitimate inputs
            c = self.best.copy()
            c.nodes = [n for n in c.nodes if name not in n.outputs]
            data = _val_to_carrier(v)
            dims = infer_labels(self.best).get(name)
            if dims is not None and len(dims) != data.ndim:
                dims = None
            t = TensorSpec(name, v.dtype, data, list(dims) if dims else [])
            if v.dtype in OPAQUE_TYPES or v.dtype in FLOAT8_TYPES:
                c.inits.append(t)
            else:
                c.inputs.append(t)
            if self._try(c):
                progress = True
                self.log(f"cut {name}")
                return True  # env is stale now
        return progress

    def pass_trim_outputs(self) -> bool:
        for i, o in enumerate(self.best.outputs):
            n = self.best.producer(o)
            if n is None:
                continue
            for inp in n.inputs:
                if not inp or self.best.tensor(inp) is not None:
                    continue
                dt = self.best.dtypes().get(inp)
                if dt in OPAQUE_TYPES or dt is None:
                    continue
                c = self.best.copy()
                c.outputs[i] = inp
                if self._try(c):
                    self.log(f"trim output -> {inp}")
                    return True
        return False

    def pass_drop_outputs(self) -> bool:
        if len(self.best.outputs) <= 1:
            return False
        for o in list(self.best.outputs):
            c = self.best.copy()
            c.outputs = [x for x in c.outputs if x != o]
            if self._try(c):
                return True
        return False

    def pass_shrink_dims(self) -> bool:
        progress = False
        labels: dict[str, int] = {}
        for t in self.best.inputs + self.best.inits:
            for ax, lab in enumerate(t.dims):
                if lab:
                    labels[lab] = t.data.shape[ax]
        for lab, n in sorted(labels.items(), key=lambda kv: -kv[1]):
            if n <= 1:
                continue
            windows = []
            if n >= 4:
                h = n // 2
                windows += [(0, h), (h, n), (0, 1), (n - 1, n)]
            windows += [(i, i + 1) for i in range(min(n, 8))]
            windows += [(0, n - 1), (1, n)]
            for lo, hi in windows:
                c = _slice_label(self.best, lab, lo, hi)
                if c is not None and self._try(c):
                    self.log(f"shrink {lab} {n}->{hi - lo}")
                    progress = True
                    break
        return progress

    def pass_shrink_blocks(self) -> bool:
        """Blocked Q/DQ: keep a single block along the blocked axis (x and its params together)."""
        for ni, n in enumerate(self.best.nodes):
            bs = n.attrs.get("block_size", 0)
            if n.op not in ("QuantizeLinear", "DequantizeLinear") or not bs:
                continue
            x = self.best.tensor(n.inputs[0])
            if x is None:
                continue
            axis = n.attrs.get("axis", 1) % x.data.ndim
            nblk = -(-x.data.shape[axis] // bs)
            if nblk <= 1:
                continue
            for k in range(nblk - 1, -1, -1):
                c = self.best.copy()
                ok = True
                for j, lo, hi in ((0, k * bs, (k + 1) * bs), (1, k, k + 1), (2, k, k + 1)):
                    if j >= len(n.inputs) or not n.inputs[j]:
                        continue
                    t = c.tensor(n.inputs[j])
                    if t is None or t.data.ndim != x.data.ndim:
                        ok = False
                        break
                    sl = [slice(None)] * t.data.ndim
                    sl[axis] = slice(lo, hi)
                    t.data = np.ascontiguousarray(t.data[tuple(sl)])
                if ok and self._try(c):
                    self.log(f"shrink blocks of node {ni} {nblk}->1")
                    return True
        return False

    def pass_simplify(self) -> bool:
        progress = False
        for t in list(self.best.inputs + self.best.inits):
            name = t.name
            cur = self.best.tensor(name)
            if cur is None or cur.data.size == 0:
                continue
            boring = _boring(cur, name in scale_names(self.best))
            flat = cur.data.ravel()
            if np.all(flat == boring):
                continue
            # whole tensor first, then ddmin chunks
            n = flat.size
            chunk = n
            while chunk >= 1:
                changed = False
                for start in range(0, n, chunk):
                    cur = self.best.tensor(name)
                    f = cur.data.ravel().copy()
                    if np.all(f[start:start + chunk] == boring):
                        continue
                    f[start:start + chunk] = boring
                    c = self.best.copy()
                    c.tensor(name).data = f.reshape(cur.data.shape)
                    if self._try(c):
                        changed = progress = True
                if chunk == 1:
                    break
                chunk = max(1, chunk // 2) if not changed else chunk
                if time.time() > self.deadline:
                    break
        return progress

    def pass_round_floats(self) -> bool:
        progress = False
        for t in list(self.best.inputs + self.best.inits):
            if t.dtype != "float32":
                continue
            for digits in (1, 2, 3):
                cur = self.best.tensor(t.name)
                f = cur.data.astype(np.float64)
                with np.errstate(all="ignore"):
                    r = np.array([float(f"{x:.{digits}g}") if np.isfinite(x) else x for x in f.ravel()]).reshape(f.shape)
                r = r.astype(np.float32)
                if np.array_equal(r, cur.data, equal_nan=True):
                    break
                c = self.best.copy()
                c.tensor(t.name).data = r
                if self._try(c):
                    progress = True
                    break
        return progress

    def run(self) -> Case:
        passes = [self.pass_drop_outputs, self.pass_cut, self.pass_trim_outputs, self.pass_shrink_dims,
                  self.pass_shrink_blocks, self.pass_simplify, self.pass_round_floats]
        while time.time() < self.deadline:
            progress = False
            for p in passes:
                while p():
                    progress = True
                    if time.time() > self.deadline:
                        break
            if not progress:
                break
        return self.best


def infer_labels(case: Case) -> dict[str, list]:
    """Propagate dimension labels through the graph (best effort)."""
    lab: dict[str, list] = {t.name: list(t.dims) for t in case.inputs + case.inits}
    for n in case.nodes:
        ins = [lab.get(i) if i else None for i in n.inputs]
        out = None
        if n.op in ("QuantizeLinear", "DequantizeLinear", "Relu", "Cast", "Identity",
                    "DynamicQuantizeLinear") and ins and ins[0]:
            out = ins[0]
        elif n.op in ("MatMul", "MatMulInteger") and len(ins) > 1 and ins[0] and ins[1]:
            out = ins[0][:-1] + ins[1][-1:]
        elif n.op == "QLinearMatMul" and ins[0] and ins[3]:
            out = ins[0][:-1] + ins[3][-1:]
        if out is not None and n.outputs and n.outputs[0]:
            lab[n.outputs[0]] = list(out)
    return lab


def _val_to_carrier(v: ref.Val) -> np.ndarray:
    if v.dtype in FLOAT8_TYPES:
        tab = ref._f8_table(v.dtype)
        out = np.zeros(v.value.shape, np.uint8)
        for idx in np.ndindex(v.value.shape):
            x = v.value[idx]
            hits = np.nonzero((tab == x) | (np.isnan(tab) & np.isnan(x)))[0]
            out[idx] = hits[0] if len(hits) else 0
        return out
    return np.asarray(v.value).astype(DTYPES[v.dtype][1])


def scale_names(case: Case) -> set:
    from .oracle import SCALE_SLOTS
    out = set()
    for n in case.nodes:
        for i in SCALE_SLOTS.get(n.op, ()):
            if i < len(n.inputs) and n.inputs[i]:
                out.add(n.inputs[i])
    return out


def _boring(t: TensorSpec, is_scale: bool = False):
    if t.dtype == "float32":
        return np.float32(1.0 if is_scale else 0.0)
    return DTYPES[t.dtype][1](0)


def _slice_label(case: Case, label: str, lo: int, hi: int) -> Case | None:
    c = case.copy()
    touched = False
    for t in c.inputs + c.inits:
        for ax, lab in enumerate(t.dims):
            if lab == label:
                sl = [slice(None)] * t.data.ndim
                sl[ax] = slice(lo, hi)
                t.data = np.ascontiguousarray(t.data[tuple(sl)])
                touched = True
    return c if touched else None


def minimize(case: Case, verdict: Verdict | None = None, budget_s: float = 120.0, log=None) -> tuple[Case, Verdict]:
    if verdict is None:
        verdict = judge(case)
    m = Minimizer(case, same_bug(verdict), budget_s, log)
    if not m.pred(m.best):  # pruning alone lost it (should not happen)
        m.best = case
    best = m.run()
    return best, judge(best)
