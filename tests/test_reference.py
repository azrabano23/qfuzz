"""Reference interpreter vs. hand-computed values from the ONNX spec."""
from fractions import Fraction

import numpy as np
import pytest

from qfuzz import reference as ref
from qfuzz.case import Case, NodeSpec, TensorSpec


def run1(op, inputs, inits, attrs=None, n_out=1, domain=""):
    """Evaluate one node; `inputs`/`inits` are (name, dtype, data) lists in node-input order."""
    names = [t[0] for t in inputs + inits]
    order = [n for n in names]
    outs = [f"y{i}" for i in range(n_out)]
    node = NodeSpec(op, order, outs, attrs or {}, domain)
    c = Case([node], [TensorSpec(*t) for t in inputs], [TensorSpec(*t) for t in inits], outs)
    env = ref.evaluate(c)
    return [env[o] for o in outs] if n_out > 1 else env[outs[0]]


def q(x, scale, zp, dtype="uint8", **attrs):
    return run1("QuantizeLinear", [("x", "float32", np.asarray(x, np.float32))],
                [("s", "float32", np.asarray(scale, np.float32)), ("z", dtype, np.asarray(zp))], attrs)


def test_quantize_round_half_to_even():
    v = q([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.4999998], 1.0, 0, "int8")
    assert v.value.tolist() == [0, 2, 2, 4, 0, -2, -2, 2]


def test_quantize_saturation_and_zero_point():
    v = q([-1000, -1, 0, 1, 1000, np.inf, -np.inf], 2.0, 128, "uint8")
    assert v.value.tolist() == [0, 128, 128, 128, 255, 255, 0]
    v = q([300.0, -300.0], 1.0, 0, "int16")
    assert v.value.tolist() == [300, -300]
    v = q([7.4, 7.6, -8.6, 100], 1.0, 0, "int4")
    assert v.value.tolist() == [7, 7, -8, 7]
    v = q([15.5, 100, -1], 1.0, 0, "uint4")
    assert v.value.tolist() == [15, 15, 0]


def test_quantize_far_from_tie_has_zero_slack_and_tie_has_one():
    v = q([1.2, 2.5], 1.0, 0, "int8")
    assert v.slack[0] == 0  # 1.2 is nowhere near a tie: must be bit exact
    assert v.slack[1] == 1  # exact tie: float32 latitude may push it either way


def test_quantize_per_axis_and_blocked():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], np.float32)
    v = q(x, [1.0, 2.0], [0, 10], "uint8", axis=0)
    assert v.value.tolist() == [[1, 2, 3, 4], [12, 13, 14, 14]]  # 5/2=2.5->2, 7/2=3.5->4
    v = q(x, [[1.0, 4.0], [2.0, 8.0]], [[0, 0], [0, 0]], "int8", axis=1, block_size=2)
    assert v.value.tolist() == [[1, 2, 1, 1], [2, 3, 1, 1]]  # 3/4=.75->1 4/4=1; 5/2=2.5->2; 7/8->1


def test_rne_exact_uses_rational_for_near_ties():
    # float64 approximation says 2.5000000001 (-> 3) but the exact value is a tie (-> 2)
    r = ref.rne_exact(np.array([2.5000000001]), lambda idx: Fraction(5, 2))
    assert r[0] == 2


def test_dequantize():
    v = run1("DequantizeLinear", [("x", "uint8", np.array([0, 128, 255], np.uint8))],
             [("s", "float32", np.float32(0.5)), ("z", "uint8", np.array(128, np.uint8))])
    assert v.value.tolist() == [-64.0, 0.0, 63.5]
    v = run1("DequantizeLinear", [], [("x", "int4", np.array([-8, 7], np.int8)), ("s", "float32", np.float32(2)),
                                      ("z", "int4", np.array(1, np.int8))])
    assert v.value.tolist() == [-18.0, 12.0]


def test_float8_rounding_matches_onnx_reference():
    x = np.array([440, 448, 450, 464, 465, 480, 500, np.inf])
    assert ref.f8_round(x, "float8e4m3fn", True).tolist() == [448.0] * 8
    got = ref.f8_round(x, "float8e4m3fn", False)
    assert got[:4].tolist() == [448.0] * 4  # 464 is a tie -> even encoding 0x7E = 448
    assert np.isnan(got[4:]).all()          # 465, 480, 500, inf -> NaN
    e5 = ref.f8_round(np.array([61439.0, 61440.0, 70000.0]), "float8e5m2", False)
    assert e5.tolist() == [57344.0, np.inf, np.inf]


def test_qlinear_matmul_hand_example():
    # ONNX spec QLinearMatMul example
    a = np.array([[208, 236, 0, 238], [3, 214, 255, 29]], np.uint8)
    b = np.array([[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]], np.uint8)
    v = run1("QLinearMatMul", [("a", "uint8", a)],
             [("as", "float32", np.float32(0.0066)), ("az", "uint8", np.array(113, np.uint8)),
              ("b", "uint8", b), ("bs", "float32", np.float32(0.00705)), ("bz", "uint8", np.array(114, np.uint8)),
              ("ys", "float32", np.float32(0.0107)), ("yz", "uint8", np.array(118, np.uint8))])
    assert v.value.tolist() == [[168, 115, 255], [1, 66, 151]]


def test_matmul_integer_and_per_column_zero_point():
    a = np.array([[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]], np.uint8)
    b = np.array([[1, 4], [2, 5], [3, 6]], np.uint8)
    v = run1("MatMulInteger", [("a", "uint8", a), ("b", "uint8", b)],
             [("az", "uint8", np.array(12, np.uint8)), ("bz", "uint8", np.array(0, np.uint8))])
    assert v.value.tolist() == [[-38, -83], [-44, -98], [-50, -113], [-56, -128]]  # spec example
    v = run1("MatMulInteger", [("a", "uint8", a), ("b", "uint8", b)],
             [("az", "uint8", np.array(0, np.uint8)), ("bz", "uint8", np.array([1, 4], np.uint8))])
    assert v.value.tolist() == (a.astype(int) @ (b.astype(int) - [1, 4])).tolist()


def test_conv_nd_matches_naive_loop():
    rng = np.random.default_rng(0)
    x = rng.integers(-5, 5, size=(1, 4, 6, 7))
    w = rng.integers(-3, 3, size=(6, 2, 3, 2))
    y = ref.conv_nd(x, w, [2, 1], [1, 2], [1, 0, 1, 1], 2)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 1)))
    oh, ow = y.shape[2:]
    naive = np.zeros_like(y)
    for m in range(6):
        g = m // 3
        for i in range(oh):
            for j in range(ow):
                for c in range(2):
                    for ki in range(3):
                        for kj in range(2):
                            naive[0, m, i, j] += xp[0, g * 2 + c, i * 2 + ki, j + kj * 2] * w[m, c, ki, kj]
    assert np.array_equal(y, naive)


def test_conv_integer_pads_with_zero_point():
    x = np.full((1, 1, 1, 1), 5, np.uint8)
    w = np.ones((1, 1, 1, 3), np.uint8)
    v = run1("ConvInteger", [("x", "uint8", x)], [("w", "uint8", w), ("xz", "uint8", np.array(5, np.uint8))],
             {"pads": [0, 1, 0, 1]})
    assert v.value.tolist() == [[[[0]]]]  # padding contributes (xz - xz) = 0


def test_dynamic_quantize_linear_spec_example():
    x = np.array([0, 2, -3, -2.5, 1.34, 0.5], np.float32)
    y, s, z = run1("DynamicQuantizeLinear", [("x", "float32", x)], [], n_out=3)
    assert abs(float(s.value) - 0.019607844) < 1e-9
    assert int(z.value) == 153
    # The spec example says 179 for x=0.5, which is what float32 division gives
    # (0.5/0.019607844 rounds to exactly 25.5 -> 26).  The exact quotient is
    # 25.4999991 -> 25 -> 178.  The reference returns the exact value and marks the
    # element with 1 LSB of slack, so both are conforming.
    assert y.value.tolist() == [153, 255, 0, 26, 221, 178]
    assert y.slack[5] == 1 and y.slack[1] == 0


def test_dynamic_quantize_all_zero_is_undefined():
    y, s, z = run1("DynamicQuantizeLinear", [("x", "float32", np.zeros(3, np.float32))], [], n_out=3)
    assert np.isinf(s.slack) and np.isinf(y.slack).all()


def test_qlinear_add_and_mul():
    a = np.array([10, 20], np.uint8)
    bb = np.array([5, 7], np.uint8)
    common = [("as", "float32", np.float32(0.5)), ("az", "uint8", np.array(0, np.uint8))]
    ins = [("a", "uint8", a)]
    inits = common + [("b", "uint8", bb), ("bs", "float32", np.float32(1.0)), ("bz", "uint8", np.array(1, np.uint8)),
                      ("cs", "float32", np.float32(2.0)), ("cz", "uint8", np.array(3, np.uint8))]
    v = run1("QLinearAdd", ins, inits, domain="com.microsoft")
    assert v.value.tolist() == [3 + 4, 3 + 8]  # (5+4)/2=4.5->4 ; (10+6)/2 = 8
    v = run1("QLinearMul", ins, inits, domain="com.microsoft")
    assert v.value.tolist() == [3 + 10, 3 + 30]  # 5*4/2=10 ; 10*6/2=30


def test_zero_scale_is_undefined():
    v = q([1.0], 0.0, 0, "uint8")
    assert np.isinf(v.slack).all()


def test_slack_propagates_through_matmul():
    # a tie in the first requantization may move by 1 LSB; a downstream integer MatMul
    # may then legitimately differ by |w| -- no more.
    x = TensorSpec("x", "float32", np.array([[2.5, 1.0]], np.float32))
    nodes = [NodeSpec("QuantizeLinear", ["x", "s", "z"], ["q"]),
             NodeSpec("MatMulInteger", ["q", "w"], ["y"])]
    c = Case(nodes, [x], [TensorSpec("s", "float32", np.float32(1.0)), TensorSpec("z", "int8", np.array(0, np.int8)),
                          TensorSpec("w", "int8", np.array([[3], [1]], np.int8))], ["y"])
    y = ref.evaluate(c)["y"]
    assert y.value.tolist() == [[2 * 3 + 1]]
    assert y.slack.tolist() == [[3.0]]
