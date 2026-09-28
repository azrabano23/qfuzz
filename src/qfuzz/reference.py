"""Exact reference interpreter for the quantized subset of ONNX.

Every tensor is evaluated to a pair ``(value, slack)``:

* ``value`` is the *ideal* result demanded by the ONNX spec: integer ops are
  computed with exact (int64) arithmetic, every quantization step
  ``round(v)`` is computed on the exact rational ``v`` with round-half-to-even
  (ties are re-checked with :class:`fractions.Fraction`, never decided by a
  float), and float tensors hold the exact result rounded once to float32.
* ``slack`` is a sound, element-wise bound on how far a *conforming*
  implementation may legitimately land from ``value``.  It models the only
  latitude the spec leaves: intermediate float arithmetic happens in float32
  (in any order; requantization multipliers may be pre-folded, e.g.
  ``a_scale*b_scale/y_scale`` as one float32), which can move a value that sits
  within a few float32 ulps of a rounding tie to the neighbouring integer.
  Slack propagates through downstream ops with interval arithmetic, so a
  1-LSB disagreement in an early requantization is allowed to be amplified by
  a later MatMul -- but only by exactly as much as the math permits.

An implementation result ``r`` is *conforming* iff ``|r - value| <= slack``.
Far from ties, slack is 0 and the reference demands bit-exact integers.

Spec ambiguities and the interpretation taken are documented in the README
("Spec ambiguities"); the short list:

1. Intermediate precision of ``x / y_scale`` is unspecified -> modelled as
   float32 latitude (a few ulps), not as exact.
2. DynamicQuantizeLinear with ``max(x) == min(x) == 0`` divides by zero in the
   spec formula -> result declared *undefined* (infinite slack) and reported
   separately.  ORT returns y_scale=1.0, the ONNX python reference 1/255.
3. Rounding for ``QLinearAdd``/``QLinearMul`` (com.microsoft, no formal spec)
   -> we use the same round-half-to-even as QuantizeLinear.
4. int32 accumulator overflow (QLinearMatMul: "accumulation may overflow if and
   only if in 32 bits") -> any result accepted (infinite slack) wherever the
   sum of |products| (+|bias|) can exceed INT32_MAX.
5. A zero scale in QuantizeLinear / requantization -> undefined (infinite slack).
6. Float8 saturate=0: ``[x] > FLT_MAX`` is read with [x] = x rounded (RNE, unbounded
   exponent) to the target mantissa, exactly like onnx.reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

from .case import DTYPES, FLOAT8_TYPES, Case, qrange

F32_EPS = 2.0**-24  # unit roundoff of float32
F32_ETA = 2.0**-149  # absolute error bound of one float32 op in the subnormal range
REL = 2.0**-20  # per-requantization latitude (~16 float32 ulps, generous)


class RefError(Exception):
    """Raised when the model is outside the reference's supported subset."""


@dataclass
class Val:
    value: np.ndarray  # float64 (float tensors) or int64 (integer tensors)
    slack: np.ndarray  # float64, same shape, >= 0
    dtype: str

    @property
    def is_int(self) -> bool:
        return self.dtype not in ("float32",) and self.dtype not in FLOAT8_TYPES


def _zeros_like(a: np.ndarray) -> np.ndarray:
    return np.zeros(np.shape(a), dtype=np.float64)


FLT_MAX = float(np.finfo(np.float32).max)


def _ovf(slack: np.ndarray, magnitude_bound: np.ndarray) -> np.ndarray:
    """Infinite slack where float32 intermediates may overflow (order-dependent inf/NaN)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(np.asarray(magnitude_bound) * (1 + 2.0**-20) >= FLT_MAX, np.inf, slack)


def _clean(slack: np.ndarray, value: np.ndarray) -> np.ndarray:
    """NaN values carry no slack; NaN slack (from inf arithmetic) means 'undefined'.

    Slack on an infinite value is kept: e.g. a DequantizeLinear whose ideal result
    overflows float32 but whose input had +/-1 LSB of latitude may legitimately
    come out finite (compare() treats inf as 2**128 for that purpose).
    """
    s = np.where(np.isnan(value), 0.0, slack)
    s = np.where(np.isnan(s), np.inf, s)
    return s


def _rel(v: np.ndarray, rel: float = None) -> np.ndarray:
    """|v| * REL on finite values only (an infinite input is exact, not 'infinitely uncertain')."""
    rel = REL if rel is None else rel
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(v), np.abs(v) * rel, 0.0)


# ----------------------------------------------------------------- float8
def _f8_table(dtype: str) -> np.ndarray:
    import ml_dtypes  # optional dep, only needed for float8

    t = {"float8e4m3fn": ml_dtypes.float8_e4m3fn, "float8e5m2": ml_dtypes.float8_e5m2}[dtype]
    return np.arange(256, dtype=np.uint8).view(t).astype(np.float64)


def f8_bits_to_f64(bits: np.ndarray, dtype: str) -> np.ndarray:
    return _f8_table(dtype)[np.asarray(bits, dtype=np.uint8)]


def f8_round(v: np.ndarray, dtype: str, saturate: bool) -> np.ndarray:
    """Round float64 -> float8 value (as float64), nearest-even, ONNX saturate rules."""
    tab = _f8_table(dtype)
    bits = np.arange(256)
    finite = np.isfinite(tab)
    fv, fb = tab[finite], bits[finite]
    order = np.lexsort((fb, fv))
    fv, fb = fv[order], fb[order]
    # unique values (drop -0 duplicate: keep +0)
    uniq, idx = np.unique(fv, return_index=True)
    ub = fb[idx]
    ub[uniq == 0] = 0
    vmax = uniq[-1]
    out = np.empty_like(v, dtype=np.float64)
    flat = v.ravel()
    o = out.ravel()
    for i, x in enumerate(flat):
        if np.isnan(x):
            o[i] = np.nan
            continue
        if abs(x) > vmax:
            # beyond max finite: RNE (with an unbounded exponent) may still round down to
            # vmax; an exact tie goes to vmax only if vmax's encoding is even (e4m3fn: 0x7E
            # yes -> 464 -> 448; e5m2: 0x7B no -> 61440 -> inf).  Matches onnx.reference.
            half_ulp = (uniq[-1] - uniq[-2]) / 2
            vmax_even = (ub[-1] & 1) == 0
            if saturate:
                o[i] = np.sign(x) * vmax
            elif np.isfinite(x) and (abs(x) < vmax + half_ulp or (abs(x) == vmax + half_ulp and vmax_even)):
                o[i] = np.sign(x) * vmax
            else:
                o[i] = np.nan if dtype == "float8e4m3fn" else np.sign(x) * np.inf
            continue
        j = np.searchsorted(uniq, x)
        if j < len(uniq) and uniq[j] == x:
            o[i] = x
            continue
        lo, hi = uniq[j - 1], uniq[j]
        if x - lo < hi - x:
            o[i] = lo
        elif hi - x < x - lo:
            o[i] = hi
        else:
            o[i] = lo if (ub[j - 1] & 1) == 0 else hi
    return out


# ---------------------------------------------------------- exact rounding
def rne_exact(q: np.ndarray, exact: Callable[[tuple], Fraction]) -> np.ndarray:
    """Round-half-to-even of the exact values approximated by float64 ``q``.

    Elements whose float64 approximation lies near a tie are recomputed with
    ``exact(index) -> Fraction`` so a float64 rounding error can never decide
    which way a tie goes.
    """
    q = np.asarray(q, dtype=np.float64)
    r = np.rint(q)
    with np.errstate(invalid="ignore"):
        frac = np.abs(q - np.floor(q) - 0.5)
        near = np.isfinite(q) & (np.abs(q) < 2.0**50) & (frac <= 1e-9 * np.maximum(1.0, np.abs(q)))
    for idx in zip(*np.nonzero(near)):
        r[idx] = float(round(exact(idx)))  # Fraction.__round__ is half-to-even
    return r


def _bcast_param(p: np.ndarray, x_shape: tuple, axis: int, block_size: int) -> np.ndarray:
    """Broadcast a scale / zero-point to x's shape (per-tensor, per-axis, blocked)."""
    p = np.asarray(p)
    nd = len(x_shape)
    if p.size == 1 and block_size == 0:
        return np.broadcast_to(p.reshape(()), x_shape)
    axis = axis % nd if nd else 0
    if block_size and block_size > 0:
        if p.ndim != nd:
            raise RefError("blocked quantization needs scale rank == input rank")
        rep = np.repeat(p, block_size, axis=axis)
        sl = [slice(None)] * nd
        sl[axis] = slice(0, x_shape[axis])
        rep = rep[tuple(sl)]
        if rep.shape != tuple(x_shape):
            raise RefError(f"blocked scale shape {p.shape} incompatible with {x_shape}")
        return rep
    if p.ndim != 1 or p.shape[0] != x_shape[axis]:
        raise RefError(f"per-axis param shape {p.shape} incompatible with {x_shape} axis {axis}")
    shape = [1] * nd
    shape[axis] = p.shape[0]
    return np.broadcast_to(p.reshape(shape), x_shape)


def _quantize_int(v: np.ndarray, vs: np.ndarray, zp: np.ndarray, dtype: str,
                  exact: Callable[[tuple], Fraction]) -> Val:
    lo_q, hi_q = qrange(dtype)
    vs = _clean(np.broadcast_to(vs, np.shape(v)), v)
    r = rne_exact(v, exact)
    with np.errstate(invalid="ignore"):
        y = np.clip(r + zp, lo_q, hi_q)
        ylo = np.clip(np.rint(v - vs) + zp, lo_q, hi_q)
        yhi = np.clip(np.rint(v + vs) + zp, lo_q, hi_q)
    y = np.where(np.isnan(y), 0, y)  # NaN input: undefined -> flagged by slack
    slack = np.maximum(np.abs(y - ylo), np.abs(yhi - y))
    slack = np.where(np.isnan(v) | np.isinf(vs), np.inf, np.where(np.isnan(slack), np.inf, slack))
    return Val(y.astype(np.int64), slack.astype(np.float64), dtype)


def _f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _fr(x) -> Fraction:
    return Fraction(float(x))


# ------------------------------------------------------------------- ops
def op_quantize(ins: list[Val | None], attrs: dict) -> list[Val]:
    x, s = ins[0], ins[1]
    zp = ins[2] if len(ins) > 2 else None
    axis = attrs.get("axis", 1)
    bs = attrs.get("block_size", 0)
    saturate = bool(attrs.get("saturate", 1))
    out_dtype = zp.dtype if zp is not None else {2: "uint8", 3: "int8"}.get(attrs.get("output_dtype", 2), None)
    if out_dtype is None:
        from .case import ELEM_TO_NAME
        out_dtype = ELEM_TO_NAME[attrs["output_dtype"]]
    shape = x.value.shape
    sc = _f64(_bcast_param(s.value, shape, axis, bs))
    z = _bcast_param(zp.value, shape, axis, bs) if zp is not None else np.zeros(shape)
    xv = _f64(x.value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = xv / sc
        vs = x.slack / np.abs(sc) + _rel(v)
        vs = np.where(sc == 0, np.inf, vs)  # division by a zero scale: undefined by the spec
    if out_dtype in FLOAT8_TYPES:
        y = f8_round(v, out_dtype, saturate)
        with np.errstate(invalid="ignore"):
            lo = f8_round(v - _clean(vs, v), out_dtype, saturate)
            hi = f8_round(v + _clean(vs, v), out_dtype, saturate)
            slack = np.maximum(np.abs(y - lo), np.abs(hi - y))
        slack = np.where(np.isnan(slack) & ~np.isnan(y), np.inf, slack)
        slack = np.where(np.isnan(y), 0.0, slack)
        return [Val(y, slack, out_dtype)]
    z = np.asarray(z, dtype=np.int64)

    def exact(idx):
        return _fr(xv[idx]) / _fr(sc[idx])

    return [_quantize_int(v, vs, z, out_dtype, exact)]


def op_dequantize(ins: list[Val | None], attrs: dict) -> list[Val]:
    x, s = ins[0], ins[1]
    zp = ins[2] if len(ins) > 2 and ins[2] is not None else None
    axis = attrs.get("axis", 1)
    bs = attrs.get("block_size", 0)
    shape = x.value.shape
    sc = _f64(_bcast_param(s.value, shape, axis, bs))
    z = _f64(_bcast_param(zp.value, shape, axis, bs)) if zp is not None else np.zeros(shape)
    xv = _f64(x.value)
    with np.errstate(invalid="ignore", over="ignore"):
        exact = (xv - z) * sc  # exact in float64: |x-z| < 2^17, scale has 24-bit mantissa
        y = exact.astype(np.float32).astype(np.float64)
        slack = x.slack * np.abs(sc) + (np.abs(xv) + np.abs(z)) * np.abs(sc) * 2 * F32_EPS + F32_ETA
    return [Val(y, _clean(slack, y), "float32")]


INT32_MAX = 2**31 - 1


def _acc_overflow(slack: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """int32 accumulation may overflow (spec is silent) -> result undefined."""
    return np.where(np.asarray(mag, dtype=np.float64) > INT32_MAX, np.inf, slack)


def _matmul_bound(a, sa, b, sb):
    return np.abs(a) @ sb + sa @ np.abs(b) + sa @ sb


def op_matmul(ins, attrs):
    a, b = ins
    av, bv = _f64(a.value), _f64(b.value)
    k = av.shape[-1]
    with np.errstate(invalid="ignore", over="ignore"):
        exact = av @ bv
        y = exact.astype(np.float32).astype(np.float64)
        mag = np.abs(av) @ np.abs(bv)
        slack = (_matmul_bound(av, a.slack, bv, b.slack)
                 + (k + 2) * F32_EPS * mag + np.abs(y) * F32_EPS + 2 * (k + 1) * F32_ETA)
        slack = _ovf(slack, mag)
    return [Val(y, _clean(slack, y), "float32")]


def _zp_rows(zp: Val | None, a_shape) -> np.ndarray:
    """Zero point for the left operand: scalar or per-row (M)."""
    if zp is None:
        return np.zeros(())
    z = np.asarray(zp.value, dtype=np.int64)
    if z.size == 1:
        return z.reshape(())
    return z.reshape(-1, 1)  # per-row


def _zp_rows_slack(zp: Val) -> np.ndarray:
    s = np.asarray(zp.slack, dtype=np.float64)
    return s.reshape(()) if s.size == 1 else s.reshape(-1, 1)


def _zp_cols_slack(zp: Val) -> np.ndarray:
    s = np.asarray(zp.slack, dtype=np.float64)
    return s.reshape(()) if s.size == 1 else s.reshape(-1)


def _zp_cols(zp: Val | None) -> np.ndarray:
    if zp is None:
        return np.zeros(())
    z = np.asarray(zp.value, dtype=np.int64)
    return z.reshape(()) if z.size == 1 else z.reshape(-1)


def op_matmul_integer(ins, attrs):
    a, b = ins[0], ins[1]
    azp = ins[2] if len(ins) > 2 else None
    bzp = ins[3] if len(ins) > 3 else None
    A = a.value.astype(np.int64) - _zp_rows(azp, a.value.shape)
    B = b.value.astype(np.int64) - _zp_cols(bzp)
    y = A @ B
    sa = a.slack + (_zp_rows_slack(azp) if azp is not None else 0)
    sb = b.slack + (_zp_cols_slack(bzp) if bzp is not None else 0)
    slack = _matmul_bound(_f64(A), np.broadcast_to(sa, A.shape), _f64(B), np.broadcast_to(sb, B.shape))
    slack = _acc_overflow(slack, np.abs(_f64(A)) @ np.abs(_f64(B)))
    return [Val(y, slack, "int32")]


def _requant(acc: np.ndarray, acc_slack: np.ndarray, mult_parts: list[np.ndarray],
             div: np.ndarray, zp: np.ndarray, dtype: str) -> Val:
    """y = saturate(round(acc * prod(mult_parts) / div) + zp) with exact ties."""
    m = np.ones(()) if not mult_parts else mult_parts[0]
    for p in mult_parts[1:]:
        m = m * p
    accf = _f64(acc)
    shape = np.broadcast_shapes(np.shape(accf), np.shape(m), np.shape(div))
    accb = np.broadcast_to(accf, shape)
    parts = [np.broadcast_to(_f64(p), shape) for p in mult_parts]
    divb = np.broadcast_to(_f64(div), shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = accb * np.broadcast_to(_f64(m), shape) / divb
        vs = np.broadcast_to(acc_slack, shape) * np.abs(v / np.where(accb == 0, 1, accb)) + np.abs(v) * REL
        # acc == 0 -> v == 0 and slack from acc_slack*|m/div|
        mm = np.abs(np.broadcast_to(_f64(m), shape) / divb)
        vs = np.where(accb == 0, np.broadcast_to(acc_slack, shape) * mm, vs)
        vs = np.where(divb == 0, np.inf, vs)  # zero output scale: undefined

    def exact(idx):
        f = Fraction(int(accb[idx]))
        for p in parts:
            f *= _fr(p[idx])
        return f / _fr(divb[idx])

    return _quantize_int(v, vs, np.broadcast_to(np.asarray(zp, dtype=np.int64), shape), dtype, exact)


def _scalar_or_vec(v: Val) -> np.ndarray:
    a = _f64(v.value)
    return a.reshape(()) if a.size == 1 else a.reshape(-1)


def op_qlinear_matmul(ins, attrs):
    a, a_s, a_zp, b, b_s, b_zp, y_s, y_zp = ins
    A = a.value.astype(np.int64) - _zp_rows(a_zp, a.value.shape)
    B = b.value.astype(np.int64) - _zp_cols(b_zp)
    acc = A @ B
    acc_slack = _matmul_bound(_f64(A), a.slack, _f64(B), b.slack)
    acc_slack = _acc_overflow(acc_slack, np.abs(_f64(A)) @ np.abs(_f64(B)))
    as_ = _scalar_or_vec(a_s)
    as_ = as_ if as_.ndim == 0 else as_.reshape(-1, 1)
    ys = _scalar_or_vec(y_s)
    ys = ys if ys.ndim == 0 else ys.reshape(-1, 1)
    yz = _zp_rows(y_zp, None)
    return [_requant(acc, acc_slack, [as_, _scalar_or_vec(b_s)], ys, yz, y_zp.dtype)]


# ------------------------------------------------------------------ conv
def _conv_attrs(x_shape, w_shape, attrs):
    nsp = len(x_shape) - 2
    if attrs.get("auto_pad", "NOTSET") not in ("NOTSET", b"NOTSET"):
        raise RefError("auto_pad not supported by reference")
    strides = list(attrs.get("strides", [1] * nsp))
    dil = list(attrs.get("dilations", [1] * nsp))
    pads = list(attrs.get("pads", [0] * (2 * nsp)))
    group = int(attrs.get("group", 1))
    return nsp, strides, dil, pads, group


def conv_nd(x: np.ndarray, w: np.ndarray, strides, dil, pads, group, pad_value=0) -> np.ndarray:
    """Direct N-d convolution (1-D or 2-D), dtype preserved (int64 exact / float64)."""
    if x.ndim == 3:
        x4 = x[:, :, None, :]
        w4 = w[:, :, None, :]
        y = conv_nd(x4, w4, [1] + list(strides), [1] + list(dil), [0, pads[0], 0, pads[1]], group, pad_value)
        return y[:, :, 0, :]
    if x.ndim != 4:
        raise RefError("only 1-D/2-D conv supported")
    n, c, h, wd = x.shape
    m, cg, kh, kw = w.shape
    pt, pl, pb, pr = pads
    xp = np.full((n, c, h + pt + pb, wd + pl + pr), pad_value, dtype=x.dtype)
    xp[:, :, pt:pt + h, pl:pl + wd] = x
    ekh = (kh - 1) * dil[0] + 1
    ekw = (kw - 1) * dil[1] + 1
    oh = (xp.shape[2] - ekh) // strides[0] + 1
    ow = (xp.shape[3] - ekw) // strides[1] + 1
    if oh <= 0 or ow <= 0:
        raise RefError("empty conv output")
    if c != cg * group or m % group:
        raise RefError("bad group config")
    mg = m // group
    out = np.zeros((n, m, oh, ow), dtype=np.result_type(x.dtype, w.dtype))
    for g in range(group):
        xs = xp[:, g * cg:(g + 1) * cg]
        ws = w[g * mg:(g + 1) * mg]
        for i in range(kh):
            for j in range(kw):
                patch = xs[:, :, i * dil[0]: i * dil[0] + strides[0] * (oh - 1) + 1: strides[0],
                           j * dil[1]: j * dil[1] + strides[1] * (ow - 1) + 1: strides[1]]
                out[:, g * mg:(g + 1) * mg] += np.einsum("nchw,mc->nmhw", patch, ws[:, :, i, j])
    return out


def _conv_bound(xa, xs, wa, ws, cfg):
    _, st, dl, pd, gp = cfg
    return (conv_nd(np.abs(xa), ws, st, dl, pd, gp) + conv_nd(xs, np.abs(wa), st, dl, pd, gp)
            + conv_nd(xs, ws, st, dl, pd, gp))


def _chan(p: np.ndarray, ndim: int) -> np.ndarray:
    """Per-output-channel param -> broadcastable against N,M,spatial..."""
    p = np.asarray(p)
    if p.size == 1:
        return p.reshape(())
    return p.reshape([1, -1] + [1] * (ndim - 2))


def op_conv(ins, attrs):
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    cfg = _conv_attrs(x.value.shape, w.value.shape, attrs)
    _, st, dl, pd, gp = cfg
    xv, wv = _f64(x.value), _f64(w.value)
    exact = conv_nd(xv, wv, st, dl, pd, gp)
    k = int(np.prod(w.value.shape[1:]))
    mag = conv_nd(np.abs(xv), np.abs(wv), st, dl, pd, gp)
    bound = _conv_bound(xv, x.slack, wv, w.slack, cfg) + (k + 3) * F32_EPS * mag + 2 * (k + 2) * F32_ETA
    if bias is not None:
        exact = exact + _chan(_f64(bias.value), exact.ndim)
        bound = bound + _chan(bias.slack, exact.ndim) + np.abs(_chan(_f64(bias.value), exact.ndim)) * F32_EPS
        mag = mag + np.abs(_chan(_f64(bias.value), exact.ndim))
    bound = _ovf(bound, mag)
    with np.errstate(over="ignore"):
        y = exact.astype(np.float32).astype(np.float64)
    return [Val(y, _clean(bound + np.abs(y) * F32_EPS, y), "float32")]


def _int_zp_chan(zp: Val | None, ndim: int, weight: bool) -> np.ndarray:
    if zp is None:
        return np.zeros(())
    z = np.asarray(zp.value, dtype=np.int64)
    if z.size == 1:
        return z.reshape(())
    if weight:
        return z.reshape([-1] + [1] * (ndim - 1))
    raise RefError("per-channel x zero point not allowed")


def op_conv_integer(ins, attrs):
    x, w = ins[0], ins[1]
    xz = ins[2] if len(ins) > 2 else None
    wz = ins[3] if len(ins) > 3 else None
    cfg = _conv_attrs(x.value.shape, w.value.shape, attrs)
    _, st, dl, pd, gp = cfg
    X = x.value.astype(np.int64) - _int_zp_chan(xz, x.value.ndim, False)
    W = w.value.astype(np.int64) - _int_zp_chan(wz, w.value.ndim, True)
    # padding is with the zero point, i.e. 0 after subtraction (spec: pad with x_zero_point)
    y = conv_nd(X, W, st, dl, pd, gp)
    slack = _conv_bound(_f64(X), x.slack, _f64(W), w.slack, cfg)
    slack = _acc_overflow(slack, conv_nd(np.abs(_f64(X)), np.abs(_f64(W)), st, dl, pd, gp))
    return [Val(y, slack, "int32")]


def op_qlinear_conv(ins, attrs):
    x, xs, xz, w, ws, wz, ys, yz = ins[:8]
    bias = ins[8] if len(ins) > 8 else None
    cfg = _conv_attrs(x.value.shape, w.value.shape, attrs)
    _, st, dl, pd, gp = cfg
    X = x.value.astype(np.int64) - _int_zp_chan(xz, x.value.ndim, False)
    W = w.value.astype(np.int64) - _int_zp_chan(wz, w.value.ndim, True)
    acc = conv_nd(X, W, st, dl, pd, gp)
    acc_slack = _conv_bound(_f64(X), x.slack, _f64(W), w.slack, cfg)
    mag = conv_nd(np.abs(_f64(X)), np.abs(_f64(W)), st, dl, pd, gp)
    if bias is not None:
        acc = acc + _chan(bias.value.astype(np.int64), acc.ndim)
        acc_slack = acc_slack + _chan(bias.slack, acc.ndim)
        mag = mag + np.abs(_chan(_f64(bias.value), acc.ndim))
    acc_slack = _acc_overflow(acc_slack, mag)
    nd = acc.ndim
    return [_requant(acc, acc_slack, [_f64(xs.value).reshape(()), _chan(_f64(ws.value), nd)],
                     _f64(ys.value).reshape(()), _chan(yz.value.astype(np.int64), nd) if yz is not None else 0,
                     yz.dtype if yz is not None else "uint8")]


def _opt(ins, i):
    return ins[i] if len(ins) > i else None


def op_qlinear_binary(kind: str):
    def f(ins, attrs):
        a, a_s, a_z, b, b_s, b_z, c_s = ins[:7]
        c_z = _opt(ins, 7)
        az = np.asarray(a_z.value, np.int64).reshape(()) if a_z is not None else 0
        bz = np.asarray(b_z.value, np.int64).reshape(()) if b_z is not None else 0
        cz = np.asarray(c_z.value, np.int64).reshape(()) if c_z is not None else 0
        A = a.value.astype(np.int64) - az
        B = b.value.astype(np.int64) - bz
        As, Bs, Cs = (_f64(t.value).reshape(()) for t in (a_s, b_s, c_s))
        A, B = np.broadcast_arrays(A, B)
        sa, sb = np.broadcast_arrays(a.slack, b.slack)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if kind == "add":
                v = (_f64(A) * As + _f64(B) * Bs) / Cs
                vs = (sa * abs(As) + sb * abs(Bs)) / abs(Cs) + (np.abs(A) * abs(As) + np.abs(B) * abs(Bs)) / abs(Cs) * REL
            else:
                v = _f64(A) * _f64(B) * As * Bs / Cs
                vs = (np.abs(A) * sb + sa * np.abs(B) + sa * sb) * abs(As * Bs / Cs) + np.abs(v) * REL

        if Cs == 0:
            vs = np.full(np.shape(v), np.inf)

        def exact(idx):
            if kind == "add":
                return (Fraction(int(A[idx])) * _fr(As) + Fraction(int(B[idx])) * _fr(Bs)) / _fr(Cs)
            return Fraction(int(A[idx]) * int(B[idx])) * _fr(As) * _fr(Bs) / _fr(Cs)

        dtype = c_z.dtype if c_z is not None else a.dtype
        return [_quantize_int(v, vs, np.full(v.shape, cz), dtype, exact)]
    return f


def op_dynamic_quantize(ins, attrs):
    x = ins[0]
    xv = x.value.astype(np.float32)
    mx = np.float32(max(0.0, float(xv.max())))
    mn = np.float32(min(0.0, float(xv.min())))
    undefined = bool(mx == mn)
    rng = np.float32(mx - mn)
    ys = np.float32(rng / np.float32(255)) if not undefined else np.float32(1.0)
    ysf = float(ys)
    # zero point: round(qmin - min/scale), saturated
    zv = np.array(0.0 - float(mn) / ysf)
    zs = np.abs(zv) * REL * 4
    zval = _quantize_int(zv.reshape(1), zs.reshape(1), np.zeros(1, np.int64), "uint8",
                         lambda idx: Fraction(0) - _fr(mn) / _fr(ys))
    v = _f64(xv) / ysf
    vs = np.abs(v) * REL * 4 + zval.slack[0]  # scale may differ by a couple of ulps
    xv64 = _f64(xv)
    yv = _quantize_int(v, vs, np.full(v.shape, zval.value[0]), "uint8",
                       lambda idx: _fr(xv64[idx]) / _fr(ys))
    s_slack = np.array(abs(ysf) * 4 * F32_EPS)
    scale = Val(np.array(ysf), s_slack, "float32")
    zp = Val(zval.value.reshape(()), zval.slack.reshape(()), "uint8")
    if undefined:
        inf = np.inf
        scale.slack = np.array(inf)
        zp.slack = np.array(inf)
        yv.slack = np.full(yv.value.shape, inf)
    return [yv, scale, zp]


def op_relu(ins, attrs):
    x = ins[0]
    return [Val(np.maximum(x.value, 0), x.slack.copy(), x.dtype)]


def op_add(ins, attrs):
    a, b = ins
    with np.errstate(invalid="ignore", over="ignore"):
        exact = _f64(a.value) + _f64(b.value)
        y = exact.astype(np.float32).astype(np.float64)
        s = np.broadcast_to(a.slack, y.shape) + np.broadcast_to(b.slack, y.shape) + np.abs(y) * F32_EPS + F32_ETA
        s = _ovf(s, np.abs(_f64(a.value)) + np.abs(_f64(b.value)))
    return [Val(y, _clean(s, y), "float32")]


def op_mul(ins, attrs):
    a, b = ins
    av, bv = _f64(a.value), _f64(b.value)
    with np.errstate(invalid="ignore", over="ignore"):
        y = (av * bv).astype(np.float32).astype(np.float64)
        s = np.abs(av) * b.slack + a.slack * np.abs(bv) + a.slack * b.slack + np.abs(y) * F32_EPS + F32_ETA
    return [Val(y, _clean(s, y), "float32")]


def op_cast(ins, attrs):
    x = ins[0]
    to = attrs["to"]
    if to != 1:
        raise RefError("Cast only to float supported")
    y = _f64(x.value).astype(np.float32).astype(np.float64)
    return [Val(y, x.slack + np.abs(y) * F32_EPS, "float32")]


def op_identity(ins, attrs):
    x = ins[0]
    return [Val(x.value.copy(), x.slack.copy(), x.dtype)]


OPS: dict[str, Callable] = {
    "QuantizeLinear": op_quantize,
    "DequantizeLinear": op_dequantize,
    "MatMul": op_matmul,
    "MatMulInteger": op_matmul_integer,
    "QLinearMatMul": op_qlinear_matmul,
    "Conv": op_conv,
    "ConvInteger": op_conv_integer,
    "QLinearConv": op_qlinear_conv,
    "QLinearAdd": op_qlinear_binary("add"),
    "QLinearMul": op_qlinear_binary("mul"),
    "DynamicQuantizeLinear": op_dynamic_quantize,
    "Relu": op_relu,
    "Add": op_add,
    "Mul": op_mul,
    "Cast": op_cast,
    "Identity": op_identity,
}


def tensor_val(t) -> Val:
    if t.dtype in FLOAT8_TYPES:
        v = f8_bits_to_f64(t.data, t.dtype)
    elif t.dtype == "float32":
        v = t.data.astype(np.float64)
    else:
        v = t.data.astype(np.int64)
    return Val(v, np.zeros(v.shape), t.dtype)


def evaluate(case: Case, keep_all: bool = False) -> dict[str, Val]:
    """Evaluate ``case`` and return {output name: Val} (all tensors if keep_all)."""
    env: dict[str, Val] = {}
    for t in case.inputs + case.inits:
        env[t.name] = tensor_val(t)
    for n in case.nodes:
        fn = OPS.get(n.op)
        if fn is None:
            raise RefError(f"op {n.op} not in reference")
        ins = [env[i] if i else None for i in n.inputs]
        outs = fn(ins, n.attrs)
        for name, v in zip(n.outputs, outs):
            if name:
                env[name] = v
    if keep_all:
        return env
    return {o: env[o] for o in case.outputs}


def to_numpy_dtype(v: Val) -> np.ndarray:
    """Reference value in the dtype ORT would return (float8 -> raw bits not supported)."""
    if v.dtype == "float32":
        return v.value.astype(np.float32)
    return v.value.astype(DTYPES[v.dtype][1])
