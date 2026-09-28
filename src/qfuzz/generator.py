"""Random generator of quantized ONNX test cases.

``generate(seed)`` is a pure function of the seed (and ``GENERATOR_VERSION``),
so a campaign only needs to store seeds to regenerate any case.

Patterns cover the standalone quantized ops, the QDQ "sandwich" patterns that
ORT's graph optimizer fuses into QLinear*/MatMulNBits/DynamicQuantizeMatMul
kernels, and a two-stage chain.  Scales, zero points and inputs are drawn
from a mixture that deliberately over-samples edge cases: exact rounding ties,
saturation boundaries, extreme / subnormal scales, boundary zero points,
infinities, and shapes at SIMD / blocking edges.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .case import DTYPES, OPAQUE_TYPES, FLOAT8_TYPES, Case, NodeSpec, TensorSpec, qrange

GENERATOR_VERSION = 2

Q8 = ["uint8", "int8"]
QALL = ["uint8", "int8", "uint16", "int16", "int4", "uint4", "float8e4m3fn", "float8e5m2"]
DIM_CHOICES = [1, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33]


class Builder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.nodes: list[NodeSpec] = []
        self.inputs: list[TensorSpec] = []
        self.inits: list[TensorSpec] = []
        self._n = 0
        self.tags: set[str] = set()

    def _name(self, p: str) -> str:
        self._n += 1
        return f"{p}{self._n}"

    def inp(self, dtype: str, data, dims=None, prefix="x") -> str:
        if dtype in OPAQUE_TYPES or dtype in FLOAT8_TYPES:
            return self.const(dtype, data, dims, prefix)
        n = self._name(prefix)
        self.inputs.append(TensorSpec(n, dtype, data, list(dims or [])))
        return n

    def const(self, dtype: str, data, dims=None, prefix="c") -> str:
        n = self._name(prefix)
        self.inits.append(TensorSpec(n, dtype, data, list(dims or [])))
        return n

    def node(self, op: str, inputs: list[str], n_out: int = 1, domain: str = "", **attrs) -> list[str]:
        outs = [self._name("t") for _ in range(n_out)]
        self.nodes.append(NodeSpec(op, inputs, outs, attrs, domain))
        return outs

    def observe(self, name: str, dtype: str) -> str:
        """Append a DequantizeLinear(scale=1, zp=0) so 4-bit/float8 outputs can be fetched."""
        if dtype not in OPAQUE_TYPES and dtype not in FLOAT8_TYPES:
            return name
        s = self.const("float32", np.array(1.0, np.float32), prefix="obs_s")
        z = self.const(dtype, np.array(0, DTYPES[dtype][1]), prefix="obs_z")
        return self.node("DequantizeLinear", [name, s, z])[0]

    def case(self, outputs: list[str], pattern: str, seed: int, **meta) -> Case:
        m = {"pattern": pattern, "seed": seed, "gen": GENERATOR_VERSION, "tags": sorted(self.tags)}
        m.update(meta)
        return Case(self.nodes, self.inputs, self.inits, outputs, m)


# ------------------------------------------------------------------ samplers
def f32(x) -> np.float32:
    return np.float32(x)


def sample_scale(b: Builder, allow_extreme: bool = True) -> np.float32:
    r = b.rng.random()
    if not allow_extreme:
        r = r * 0.7
    if r < 0.45:
        s = 10 ** b.rng.uniform(-4, 0)
    elif r < 0.60:
        s = 2.0 ** int(b.rng.integers(-14, 6))
        b.tags.add("scale-pow2")
    elif r < 0.70:
        s = 10 ** b.rng.uniform(0, 4)
    elif r < 0.78:
        s = 10 ** b.rng.uniform(-37.5, -30)
        b.tags.add("scale-tiny")
    elif r < 0.84:
        s = 10 ** b.rng.uniform(-44.5, -38.5)  # float32 subnormal range
        b.tags.add("scale-subnormal")
    elif r < 0.93:
        s = 10 ** b.rng.uniform(15, 38)
        b.tags.add("scale-huge")
    else:
        s = float(np.nextafter(np.float32(1.0), np.float32(2.0))) * 10 ** b.rng.uniform(-3, 0)
    s = f32(s)
    if s == 0 or not np.isfinite(s):
        s = f32(np.finfo(np.float32).tiny) if s == 0 else f32(1e38)
    return s


def sample_scales(b: Builder, n: int, allow_extreme=True) -> np.ndarray:
    if b.rng.random() < 0.5:
        base = sample_scale(b, allow_extreme)
        return (base * (2.0 ** b.rng.uniform(-2, 2, size=n))).astype(np.float32).clip(min=np.float32(1.4e-45))
    return np.array([sample_scale(b, allow_extreme) for _ in range(n)], np.float32)


def sample_zp(b: Builder, dtype: str, shape=()) -> np.ndarray:
    carrier = DTYPES[dtype][1]
    if dtype in FLOAT8_TYPES:
        return np.zeros(shape, carrier)
    lo, hi = qrange(dtype)
    r = b.rng.random()
    if r < 0.45:
        z = b.rng.integers(lo, hi + 1, size=shape)
    elif r < 0.75:
        b.tags.add("zp-boundary")
        z = b.rng.choice([lo, hi, lo + 1, hi - 1], size=shape)
    else:
        z = np.full(shape, 0 if lo < 0 else (lo + hi + 1) // 2)
    return np.asarray(z).astype(carrier)


def sample_qdata(b: Builder, dtype: str, shape) -> np.ndarray:
    carrier = DTYPES[dtype][1]
    if dtype in FLOAT8_TYPES:
        bits = b.rng.integers(0, 256, size=shape)
        if dtype == "float8e4m3fn":
            bits = np.where((bits & 0x7F) == 0x7F, 0, bits)  # avoid NaN encodings
        else:
            bits = np.where((bits & 0x7C) == 0x7C, 0, bits)  # avoid inf/NaN encodings
        return bits.astype(carrier)
    lo, hi = qrange(dtype)
    x = b.rng.integers(lo, hi + 1, size=shape)
    if b.rng.random() < 0.3:
        b.tags.add("q-boundary")
        mask = b.rng.random(size=shape) < 0.4
        x = np.where(mask, b.rng.choice([lo, hi], size=shape), x)
    return x.astype(carrier)


def sample_float_for(b: Builder, shape, scale: np.ndarray, zp: np.ndarray, dtype: str) -> np.ndarray:
    """Float inputs that land around the representable range of (scale, zp, dtype)."""
    scale = np.broadcast_to(np.asarray(scale, np.float64), shape)
    if dtype in FLOAT8_TYPES:
        mx = 448.0 if dtype == "float8e4m3fn" else 57344.0
        v = b.rng.choice([-1, 1], size=shape) * mx * 10 ** b.rng.uniform(-6, 0.2, size=shape)
        x = (v * scale).astype(np.float32)
    else:
        lo, hi = qrange(dtype)
        zpb = np.broadcast_to(np.asarray(zp, np.float64), shape)
        u = b.rng.uniform(lo - zpb - 3, hi - zpb + 3)
        r = b.rng.random()
        if r < 0.35:  # exact ties k + 0.5
            b.tags.add("ties")
            u = np.floor(u) + 0.5
        elif r < 0.45:  # saturation boundaries
            b.tags.add("sat-boundary")
            u = b.rng.choice([lo, hi, lo - 0.5, hi + 0.5, lo - 1, hi + 1], size=shape) - zpb
        x = (u * scale).astype(np.float32)
    with np.errstate(over="ignore"):
        if b.rng.random() < 0.1:
            b.tags.add("specials")
            mask = b.rng.random(size=shape) < 0.2
            special = b.rng.choice(np.array([np.inf, -np.inf, 0.0, -0.0, 3.0e38, -3.0e38], np.float32), size=shape)
            x = np.where(mask, special, x)
    return np.asarray(x, np.float32)


def dim(b: Builder, big: bool = False) -> int:
    if big and b.rng.random() < 0.25:
        return int(b.rng.choice([63, 64, 65, 127, 128, 129, 255, 256, 257, 512]))
    return int(b.rng.choice(DIM_CHOICES))


def choose_quant_mode(b: Builder, x_shape, dtype: str, allow_block=True):
    """Returns (attrs, param_shape, param_dims)."""
    nd = len(x_shape)
    r = b.rng.random()
    if r < 0.5 or nd == 0:
        return {}, (), []
    axis = int(b.rng.integers(0, nd))
    if r < 0.8 or not allow_block or dtype in FLOAT8_TYPES:
        b.tags.add("per-axis")
        attrs = {"axis": axis} if (axis != 1 or b.rng.random() < 0.5) else {}
        if nd == 1 and "axis" not in attrs:
            attrs = {"axis": 0}
        return attrs, (x_shape[axis],), [f"d{axis}"]
    b.tags.add("blocked")
    bs = int(b.rng.choice([1, 2, 3, 4, 8, 16, 32]))
    pshape = list(x_shape)
    pshape[axis] = math.ceil(x_shape[axis] / bs)
    return {"axis": axis, "block_size": bs}, tuple(pshape), None


def _block_dims(shape, axis):
    """Blocked quantization: every axis but the blocked one can be shrunk jointly."""
    d = [f"d{i}" for i in range(len(shape))]
    d[axis] = None
    return d


def _dims(shape):
    return [f"d{i}" for i in range(len(shape))]


def rand_shape(b: Builder, rank=None) -> tuple:
    rank = int(b.rng.integers(1, 5)) if rank is None else rank
    return tuple(dim(b) for _ in range(rank))


# ------------------------------------------------------------------ patterns
def p_quantize(b: Builder, seed: int) -> Case:
    dtype = str(b.rng.choice(QALL))
    shape = rand_shape(b)
    attrs, pshape, pdims = choose_quant_mode(b, shape, dtype)
    blocked = "block_size" in attrs
    scale = sample_scales(b, int(np.prod(pshape)) if pshape else 1).reshape(pshape)
    zp = sample_zp(b, dtype, pshape)
    x = sample_float_for(b, shape, _expand(scale, shape, attrs), _expand(zp.astype(np.float64), shape, attrs)
                         if dtype not in FLOAT8_TYPES else 0, dtype)
    bd = _block_dims(shape, attrs["axis"]) if blocked else None
    dims = bd if blocked else _dims(shape)
    xi = b.inp("float32", x, dims)
    s = b.const("float32", scale, bd if blocked else pdims)
    ins = [xi, s]
    if dtype == "uint8" and b.rng.random() < 0.15:
        b.tags.add("no-zp")
    else:
        ins.append(b.const(dtype, zp, bd if blocked else pdims))
    if dtype in FLOAT8_TYPES and b.rng.random() < 0.5:
        attrs = dict(attrs, saturate=0)
        b.tags.add("saturate0")
    y = b.node("QuantizeLinear", ins, **attrs)[0]
    y = b.observe(y, dtype)
    return b.case([y], "quantize", seed, dtype=dtype)


def _expand(p, shape, attrs):
    from .reference import _bcast_param
    p = np.asarray(p)
    if p.size == 1 and not attrs.get("block_size"):
        return p.reshape(())
    return _bcast_param(p, shape, attrs.get("axis", 1), attrs.get("block_size", 0))


def p_dequantize(b: Builder, seed: int) -> Case:
    dtype = str(b.rng.choice(QALL + ["int32"]))
    shape = rand_shape(b)
    attrs, pshape, pdims = choose_quant_mode(b, shape, dtype, allow_block=dtype != "int32")
    blocked = "block_size" in attrs
    scale = sample_scales(b, int(np.prod(pshape)) if pshape else 1).reshape(pshape)
    x = sample_qdata(b, dtype, shape) if dtype != "int32" else b.rng.integers(-(2**31), 2**31, size=shape).astype(np.int32)
    bd = _block_dims(shape, attrs["axis"]) if blocked else None
    xi = b.inp(dtype, x, bd if blocked else _dims(shape))
    ins = [xi, b.const("float32", scale, bd if blocked else pdims)]
    if dtype != "int32" and b.rng.random() < 0.85:
        ins.append(b.const(dtype, sample_zp(b, dtype, pshape), bd if blocked else pdims))
    y = b.node("DequantizeLinear", ins, **attrs)[0]
    return b.case([y], "dequantize", seed, dtype=dtype)


def p_qdq(b: Builder, seed: int) -> Case:
    """x -> Q -> DQ (-> Q -> DQ), exercising QDQ pair elimination / propagation."""
    dtype = str(b.rng.choice(["uint8", "int8", "uint16", "int16"]))
    shape = rand_shape(b)
    s1, z1 = sample_scale(b), sample_zp(b, dtype)
    x = sample_float_for(b, shape, s1, z1, dtype)
    cur = b.inp("float32", x, _dims(shape))
    stages = int(b.rng.integers(1, 3))
    same = b.rng.random() < 0.5
    for i in range(stages):
        if i > 0 and not same:
            s1, z1 = sample_scale(b), sample_zp(b, dtype)
        sc, zc = b.const("float32", np.array(s1)), b.const(dtype, np.array(z1))
        q = b.node("QuantizeLinear", [cur, sc, zc])[0]
        sc2, zc2 = (sc, zc) if b.rng.random() < 0.8 else (b.const("float32", np.array(sample_scale(b))), b.const(dtype, sample_zp(b, dtype)))
        cur = b.node("DequantizeLinear", [q, sc2, zc2])[0]
    if b.rng.random() < 0.3:
        cur = b.node("Relu", [cur])[0]
        b.tags.add("relu")
    return b.case([cur], "qdq", seed, dtype=dtype, stages=stages)


# (a, b, y) type combos ORT's CPU EP implements for QLinearMatMul / QLinearConv; the
# spec allows all 8 -- the rest are sampled rarely so they don't swamp the budget.
ORT_Q8_COMBOS = [("uint8", "uint8", "uint8"), ("uint8", "int8", "uint8"), ("int8", "int8", "int8")]


def pick_combo(b: Builder) -> tuple[str, str, str]:
    if b.rng.random() < 0.85:
        return ORT_Q8_COMBOS[int(b.rng.integers(0, len(ORT_Q8_COMBOS)))]
    b.tags.add("rare-type-combo")
    return tuple(str(b.rng.choice(Q8)) for _ in range(3))


def _mm_scales(b: Builder, a_s, b_s, k: int):
    """Output scale that makes outputs land in range most of the time."""
    base = float(np.max(a_s)) * float(np.max(b_s)) * math.sqrt(k) * 40.0 / 127.0
    if b.rng.random() < 0.2:
        return sample_scale(b)
    return f32(base * 10 ** b.rng.uniform(-1, 1)) if np.isfinite(base) and base > 0 else sample_scale(b)


def _mm_operands(b: Builder, allow_percol=True):
    batch = () if b.rng.random() < 0.8 else (dim(b),)
    m, k, n = dim(b), dim(b, big=True), dim(b)
    adt, bdt, ydt = pick_combo(b)
    a = sample_qdata(b, adt, batch + (m, k))
    w = sample_qdata(b, bdt, (k, n))
    a_s = sample_scale(b)
    per_col = allow_percol and b.rng.random() < 0.4
    b_s = sample_scales(b, n) if per_col else np.array(sample_scale(b), np.float32)
    a_z = sample_zp(b, adt)
    b_z = sample_zp(b, bdt, (n,)) if per_col else sample_zp(b, bdt)
    if per_col:
        b.tags.add("per-col")
    adims = (["B"] if batch else []) + ["M", "K"]
    return dict(batch=batch, m=m, k=k, n=n, adt=adt, bdt=bdt, ydt=ydt, a=a, w=w, a_s=a_s, b_s=b_s, a_z=a_z,
                b_z=b_z, per_col=per_col, adims=adims)


def p_dq_matmul_q(b: Builder, seed: int) -> Case:
    o = _mm_operands(b)
    ydt = o["ydt"]
    ai = b.inp(o["adt"], o["a"], o["adims"], "a")
    wi = b.const(o["bdt"], o["w"], ["K", "N"], "w")
    a_dq = b.node("DequantizeLinear", [ai, b.const("float32", np.array(o["a_s"])), b.const(o["adt"], np.array(o["a_z"]))])[0]
    pd = ["N"] if o["per_col"] else []
    w_dq = b.node("DequantizeLinear", [wi, b.const("float32", o["b_s"], pd), b.const(o["bdt"], o["b_z"], pd)],
                  **({"axis": 1} if o["per_col"] else {}))[0]
    mm = b.node("MatMul", [a_dq, w_dq])[0]
    y_s, y_z = _mm_scales(b, o["a_s"], o["b_s"], o["k"]), sample_zp(b, ydt)
    y = b.node("QuantizeLinear", [mm, b.const("float32", np.array(y_s)), b.const(ydt, np.array(y_z))])[0]
    return b.case([y], "dq_matmul_q", seed)


def p_qlinear_matmul(b: Builder, seed: int) -> Case:
    o = _mm_operands(b)
    ydt = o["ydt"]
    a_s = np.array(o["a_s"])
    a_z = np.array(o["a_z"])
    adims_p = []
    if b.rng.random() < 0.05 and not o["batch"]:
        b.tags.add("per-row-a")
        a_s = sample_scales(b, o["m"])
        a_z = sample_zp(b, o["adt"], (o["m"],))
        adims_p = ["M"]
    pd = ["N"] if o["per_col"] else []
    y_s, y_z = _mm_scales(b, a_s, o["b_s"], o["k"]), sample_zp(b, ydt)
    ins = [b.inp(o["adt"], o["a"], o["adims"], "a"), b.const("float32", a_s, adims_p), b.const(o["adt"], a_z, adims_p),
           b.inp(o["bdt"], o["w"], ["K", "N"], "b") if b.rng.random() < 0.3 else b.const(o["bdt"], o["w"], ["K", "N"], "w"),
           b.const("float32", o["b_s"], pd), b.const(o["bdt"], o["b_z"], pd),
           b.const("float32", np.array(y_s)), b.const(ydt, np.array(y_z))]
    y = b.node("QLinearMatMul", ins)[0]
    return b.case([y], "qlinear_matmul", seed)


def p_matmul_integer(b: Builder, seed: int) -> Case:
    o = _mm_operands(b)
    ins = [b.inp(o["adt"], o["a"], o["adims"], "a"),
           b.inp(o["bdt"], o["w"], ["K", "N"], "b") if b.rng.random() < 0.3 else b.const(o["bdt"], o["w"], ["K", "N"], "w")]
    r = b.rng.random()
    if r < 0.8:
        a_z, adp = np.array(o["a_z"]), []
        if b.rng.random() < 0.05 and not o["batch"]:
            b.tags.add("per-row-a")
            a_z, adp = sample_zp(b, o["adt"], (o["m"],)), ["M"]
        ins.append(b.const(o["adt"], a_z, adp))
        if r < 0.6:
            pd = ["N"] if o["per_col"] else []
            ins.append(b.const(o["bdt"], o["b_z"], pd))
    y = b.node("MatMulInteger", ins)[0]
    return b.case([y], "matmul_integer", seed)


def _conv_operands(b: Builder):
    rank = 2 if b.rng.random() < 0.85 else 1
    n = int(b.rng.choice([1, 1, 2]))
    group = int(b.rng.choice([1, 1, 1, 2, 4]))
    cg = int(b.rng.choice([1, 2, 3, 4, 8, 16]))
    mg = int(b.rng.choice([1, 2, 3, 4, 8, 16]))
    if b.rng.random() < 0.15:  # depthwise
        cg, mg, group = 1, 1, int(b.rng.choice([3, 8, 16, 32]))
        b.tags.add("depthwise")
    c, m = cg * group, mg * group
    ks = [int(b.rng.choice([1, 1, 2, 3, 3, 5])) for _ in range(rank)]
    strides = [int(b.rng.choice([1, 1, 2, 3])) for _ in range(rank)]
    dil = [int(b.rng.choice([1, 1, 1, 2])) for _ in range(rank)]
    pads = [int(b.rng.integers(0, k)) if b.rng.random() < 0.5 else 0 for k in ks] * 2
    spatial = [int(max((k - 1) * d + 1, b.rng.choice([1, 2, 3, 5, 7, 8, 9, 14, 16]))) for k, d in zip(ks, dil)]
    xdt, wdt, ydt = pick_combo(b)
    x = sample_qdata(b, xdt, (n, c, *spatial))
    w = sample_qdata(b, wdt, (m, cg, *ks))
    attrs = {"kernel_shape": ks}
    if any(s != 1 for s in strides):
        attrs["strides"] = strides
    if any(d != 1 for d in dil):
        attrs["dilations"] = dil
    if any(pads):
        attrs["pads"] = pads
    if group != 1:
        attrs["group"] = group
    per_ch = b.rng.random() < 0.4
    if per_ch:
        b.tags.add("per-channel")
    x_s = sample_scale(b)
    w_s = sample_scales(b, m) if per_ch else np.array(sample_scale(b), np.float32)
    x_z = sample_zp(b, xdt)
    w_z = np.zeros((m,) if per_ch else (), DTYPES[wdt][1]) if (wdt == "int8" and b.rng.random() < 0.7) else sample_zp(b, wdt, (m,) if per_ch else ())
    xdims = ["N", "C" if group == 1 else None] + [f"S{i}" for i in range(rank)]
    wdims = ["M", "C"] + [None] * rank if group == 1 else None
    chdims = ["M"] if (per_ch and group == 1) else None
    return dict(ydt=ydt, wdims=wdims, chdims=chdims, x=x, w=w, xdt=xdt, wdt=wdt, attrs=attrs, x_s=x_s, w_s=w_s, x_z=x_z, w_z=w_z, per_ch=per_ch,
                m=m, k=cg * int(np.prod(ks)), xdims=xdims, group=group)


def p_qlinear_conv(b: Builder, seed: int) -> Case:
    o = _conv_operands(b)
    ydt = o["ydt"]
    y_s, y_z = _mm_scales(b, o["x_s"], o["w_s"], o["k"]), sample_zp(b, ydt)
    pd = ["M"] if o["per_ch"] else []
    ins = [b.inp(o["xdt"], o["x"], o["xdims"], "x"), b.const("float32", np.array(o["x_s"])), b.const(o["xdt"], np.array(o["x_z"])),
           b.const(o["wdt"], o["w"], o["wdims"], "w"), b.const("float32", o["w_s"], o["chdims"]), b.const(o["wdt"], o["w_z"], o["chdims"]),
           b.const("float32", np.array(y_s)), b.const(ydt, np.array(y_z))]
    if b.rng.random() < 0.6:
        bias = b.rng.integers(-5000, 5000, size=(o["m"],))
        if b.rng.random() < 0.1:
            bias = b.rng.choice([-(2**31), 2**31 - 1, -(2**24), 2**24], size=(o["m"],))
            b.tags.add("bias-extreme")
        ins.append(b.const("int32", bias.astype(np.int32), ["M"] if o["group"] == 1 else None))
    y = b.node("QLinearConv", ins, **o["attrs"])[0]
    return b.case([y], "qlinear_conv", seed)


def p_conv_integer(b: Builder, seed: int) -> Case:
    o = _conv_operands(b)
    ins = [b.inp(o["xdt"], o["x"], o["xdims"], "x"), b.const(o["wdt"], o["w"], o["wdims"], "w")]
    r = b.rng.random()
    if r < 0.8:
        ins.append(b.const(o["xdt"], np.array(o["x_z"])))
        if r < 0.6:
            ins.append(b.const(o["wdt"], o["w_z"], o["chdims"]))
    y = b.node("ConvInteger", ins, **o["attrs"])[0]
    return b.case([y], "conv_integer", seed)


def p_dq_conv_q(b: Builder, seed: int) -> Case:
    o = _conv_operands(b)
    ydt = o["ydt"]
    xi = b.inp(o["xdt"], o["x"], o["xdims"], "x")
    x_dq = b.node("DequantizeLinear", [xi, b.const("float32", np.array(o["x_s"])), b.const(o["xdt"], np.array(o["x_z"]))])[0]
    w_dq = b.node("DequantizeLinear", [b.const(o["wdt"], o["w"], o["wdims"], "w"), b.const("float32", o["w_s"], o["chdims"]),
                                       b.const(o["wdt"], o["w_z"], o["chdims"])], **({"axis": 0} if o["per_ch"] else {}))[0]
    ins = [x_dq, w_dq]
    if b.rng.random() < 0.6:
        bias = b.rng.integers(-5000, 5000, size=(o["m"],)).astype(np.int32)
        bs = (np.float32(o["x_s"]) * np.broadcast_to(o["w_s"], (o["m"],))).astype(np.float32)
        bs_attrs = {"axis": 0}
        if not o["per_ch"]:
            bs, bs_attrs = np.array(bs[0]), {}
        bdim = ["M"] if o["group"] == 1 else None
        ins.append(b.node("DequantizeLinear", [b.const("int32", bias, bdim),
                                               b.const("float32", bs, bdim if o["per_ch"] else None)], **bs_attrs)[0])
    conv = b.node("Conv", ins, **o["attrs"])[0]
    y_s, y_z = _mm_scales(b, o["x_s"], o["w_s"], o["k"]), sample_zp(b, ydt)
    if b.rng.random() < 0.2:
        conv = b.node("Relu", [conv])[0]
        b.tags.add("relu")
    y = b.node("QuantizeLinear", [conv, b.const("float32", np.array(y_s)), b.const(ydt, np.array(y_z))])[0]
    return b.case([y], "dq_conv_q", seed)


def _binary_shapes(b: Builder):
    """Two broadcast-compatible shapes plus their dimension labels."""
    shape = rand_shape(b, int(b.rng.integers(1, 4)))
    lab = _dims(shape)
    r = b.rng.random()
    if r < 0.6:
        return shape, shape, lab, lab
    if r < 0.8:
        return shape, (shape[-1],), lab, lab[-1:]
    return shape, (), lab, []


def p_qlinear_binary(b: Builder, seed: int) -> Case:
    kind = str(b.rng.choice(["QLinearAdd", "QLinearMul"]))
    dt = str(b.rng.choice(Q8))
    sa, sb, la, lb = _binary_shapes(b)
    if b.rng.random() < 0.5:
        sa, sb, la, lb = sb, sa, lb, la
    a, bb = sample_qdata(b, dt, sa), sample_qdata(b, dt, sb)
    a_s, b_s = sample_scale(b), sample_scale(b)
    c_s = f32(a_s + b_s) * f32(10 ** b.rng.uniform(-0.5, 0.5)) if kind == "QLinearAdd" else f32(a_s * b_s * 64 * 10 ** b.rng.uniform(-1, 1))
    if not np.isfinite(c_s) or c_s == 0:
        c_s = sample_scale(b)
    ins = [b.inp(dt, a, la, "a"), b.const("float32", np.array(a_s)), b.const(dt, sample_zp(b, dt)),
           b.inp(dt, bb, lb, "b") if b.rng.random() < 0.7 else b.const(dt, bb, lb), b.const("float32", np.array(b_s)),
           b.const(dt, sample_zp(b, dt)), b.const("float32", np.array(c_s)), b.const(dt, sample_zp(b, dt))]
    y = b.node(kind, ins, domain="com.microsoft")[0]
    return b.case([y], "qlinear_binary", seed, op=kind)


def p_dq_binary_q(b: Builder, seed: int) -> Case:
    kind = str(b.rng.choice(["Add", "Mul"]))
    dt = str(b.rng.choice(Q8))
    sa, sb, la, lb = _binary_shapes(b)
    a, bb = sample_qdata(b, dt, sa), sample_qdata(b, dt, sb)
    a_s, b_s = sample_scale(b), sample_scale(b)
    c_s = f32(a_s + b_s) if kind == "Add" else f32(a_s * b_s * 64)
    if not np.isfinite(c_s) or c_s == 0 or b.rng.random() < 0.2:
        c_s = sample_scale(b)
    ad = b.node("DequantizeLinear", [b.inp(dt, a, la, "a"), b.const("float32", np.array(a_s)), b.const(dt, sample_zp(b, dt))])[0]
    bd = b.node("DequantizeLinear", [b.inp(dt, bb, lb, "b"), b.const("float32", np.array(b_s)), b.const(dt, sample_zp(b, dt))])[0]
    s = b.node(kind, [ad, bd])[0]
    y = b.node("QuantizeLinear", [s, b.const("float32", np.array(c_s)), b.const(dt, sample_zp(b, dt))])[0]
    return b.case([y], "dq_binary_q", seed, op=kind)


def p_dq_relu_q(b: Builder, seed: int) -> Case:
    dt = str(b.rng.choice(Q8))
    shape = rand_shape(b)
    s = sample_scale(b)
    z = sample_zp(b, dt)
    x = sample_qdata(b, dt, shape)
    d = b.node("DequantizeLinear", [b.inp(dt, x, _dims(shape)), b.const("float32", np.array(s)), b.const(dt, z)])[0]
    r = b.node("Relu", [d])[0]
    s2, z2 = (s, z) if b.rng.random() < 0.6 else (sample_scale(b), sample_zp(b, dt))
    y = b.node("QuantizeLinear", [r, b.const("float32", np.array(s2)), b.const(dt, np.asarray(z2))])[0]
    return b.case([y], "dq_relu_q", seed)


def _dyn_input(b: Builder, shape) -> np.ndarray:
    r = b.rng.random()
    mag = 10 ** b.rng.uniform(-3, 3)
    if r < 0.05:
        b.tags.add("all-zero")
        return np.zeros(shape, np.float32)
    if r < 0.2:
        b.tags.add("one-signed")
        return (np.abs(b.rng.normal(size=shape)) * mag * b.rng.choice([-1, 1])).astype(np.float32)
    if r < 0.3:
        b.tags.add("ties")
        k = b.rng.integers(-128, 128, size=shape) + 0.5
        return (k * f32(2.0 ** int(b.rng.integers(-8, 4)))).astype(np.float32)
    if r < 0.35:
        b.tags.add("extreme-range")
        return (b.rng.normal(size=shape) * 10 ** b.rng.uniform(30, 37)).astype(np.float32)
    return (b.rng.normal(size=shape) * mag).astype(np.float32)


def p_dynamic_quantize(b: Builder, seed: int) -> Case:
    shape = rand_shape(b)
    xi = b.inp("float32", _dyn_input(b, shape), _dims(shape))
    y, s, z = b.node("DynamicQuantizeLinear", [xi], n_out=3)
    return b.case([y, s, z], "dynamic_quantize", seed)


def p_dyn_matmul(b: Builder, seed: int) -> Case:
    """DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(scale) : fused by ORT into DynamicQuantizeMatMul."""
    m, k, n = dim(b), dim(b, big=True), dim(b)
    a = _dyn_input(b, (m, k))
    wdt = str(b.rng.choice(Q8))
    w = sample_qdata(b, wdt, (k, n))
    w_s = sample_scale(b, allow_extreme=False)
    w_z = sample_zp(b, wdt) if b.rng.random() < 0.5 else np.zeros((), DTYPES[wdt][1])
    y, s, z = b.node("DynamicQuantizeLinear", [b.inp("float32", a, None, "a")], n_out=3)
    mmi = b.node("MatMulInteger", [y, b.const(wdt, w, None, "w"), z, b.const(wdt, w_z)])[0]
    cf = b.node("Cast", [mmi], to=1)[0]
    sc = b.node("Mul", [s, b.const("float32", np.array(w_s))])[0]
    out = b.node("Mul", [cf, sc])[0]
    return b.case([out], "dyn_matmul", seed)


def p_dq_matmul_nbits(b: Builder, seed: int) -> Case:
    """float A @ DQ(int4 blocked W): ORT fuses DQ+MatMul into MatMulNBits."""
    bs = int(b.rng.choice([16, 32, 64, 128]))
    nblk = int(b.rng.choice([1, 1, 2, 3, 4]))
    k = bs * nblk
    m, n = dim(b), dim(b)
    wdt = str(b.rng.choice(["int4", "uint4"]))
    w = sample_qdata(b, wdt, (k, n))
    scale = sample_scales(b, nblk * n, allow_extreme=b.rng.random() < 0.3).reshape(nblk, n)
    a = (b.rng.normal(size=(m, k)) * 10 ** b.rng.uniform(-2, 2)).astype(np.float32)
    ins = [b.const(wdt, w, None, "w"), b.const("float32", scale)]
    if b.rng.random() < 0.7:
        ins.append(b.const(wdt, sample_zp(b, wdt, (nblk, n))))
    wd = b.node("DequantizeLinear", ins, axis=0, block_size=bs)[0]
    y = b.node("MatMul", [b.inp("float32", a, ["M", None], "a"), wd])[0]
    return b.case([y], "dq_matmul_nbits", seed, block_size=bs)


def p_chain(b: Builder, seed: int) -> Case:
    """Two DQ->MatMul->Q stages back to back."""
    m = dim(b)
    dt = str(b.rng.choice(Q8))
    ks = [dim(b), dim(b), dim(b)]
    s0, z0 = sample_scale(b, False), sample_zp(b, dt)
    cur_q = b.inp(dt, sample_qdata(b, dt, (m, ks[0])), ["M", "K0"], "a")
    for i in range(2):
        wdt = str(b.rng.choice(Q8))
        w = sample_qdata(b, wdt, (ks[i], ks[i + 1]))
        ws, wz = sample_scale(b, False), sample_zp(b, wdt)
        ad = b.node("DequantizeLinear", [cur_q, b.const("float32", np.array(s0)), b.const(dt, np.array(z0))])[0]
        wd = b.node("DequantizeLinear", [b.const(wdt, w, [f"K{i}", f"K{i+1}"]), b.const("float32", np.array(ws)), b.const(wdt, np.array(wz))])[0]
        mm = b.node("MatMul", [ad, wd])[0]
        s0, z0 = _mm_scales(b, s0, ws, ks[i]), sample_zp(b, dt)
        cur_q = b.node("QuantizeLinear", [mm, b.const("float32", np.array(s0)), b.const(dt, np.array(z0))])[0]
    return b.case([cur_q], "chain", seed)


PATTERNS: dict[str, tuple[float, Callable[[Builder, int], Case]]] = {
    "quantize": (3.0, p_quantize),
    "dequantize": (2.0, p_dequantize),
    "qdq": (1.5, p_qdq),
    "dq_matmul_q": (2.0, p_dq_matmul_q),
    "qlinear_matmul": (2.0, p_qlinear_matmul),
    "matmul_integer": (1.5, p_matmul_integer),
    "qlinear_conv": (1.5, p_qlinear_conv),
    "conv_integer": (1.0, p_conv_integer),
    "dq_conv_q": (1.5, p_dq_conv_q),
    "qlinear_binary": (1.5, p_qlinear_binary),
    "dq_binary_q": (1.5, p_dq_binary_q),
    "dq_relu_q": (1.0, p_dq_relu_q),
    "dynamic_quantize": (1.5, p_dynamic_quantize),
    "dyn_matmul": (1.0, p_dyn_matmul),
    "dq_matmul_nbits": (1.0, p_dq_matmul_nbits),
    "chain": (1.0, p_chain),
}


def generate(seed: int, pattern: str | None = None) -> Case:
    rng = np.random.default_rng([GENERATOR_VERSION, seed])
    names = list(PATTERNS)
    if pattern is None:
        w = np.array([PATTERNS[n][0] for n in names])
        pattern = names[int(rng.choice(len(names), p=w / w.sum()))]
    b = Builder(rng)
    return PATTERNS[pattern][1](b, seed)
