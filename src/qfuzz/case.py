"""Test-case representation.

A :class:`Case` is a tiny, JSON-serialisable description of an ONNX graph plus
concrete values for every graph input and initializer.  Everything else
(the ONNX model, ORT feeds, the reference evaluation, the standalone repro
script) is derived from it, which is what makes delta-debugging simple: the
minimizer edits a ``Case`` and re-derives the rest.

Each tensor carries optional *dimension labels* (e.g. ``["M", "K"]``).  The
minimizer shrinks a label by slicing every tensor axis that carries it, which
keeps coupled dimensions (the ``K`` of ``A`` and ``B`` in a MatMul) in sync
without op-specific code.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper

# name -> (onnx elem type, numpy carrier dtype, integer range or None)
DTYPES: dict[str, tuple[int, Any, tuple[int, int] | None]] = {
    "float32": (TP.FLOAT, np.float32, None),
    "uint8": (TP.UINT8, np.uint8, (0, 255)),
    "int8": (TP.INT8, np.int8, (-128, 127)),
    "uint16": (TP.UINT16, np.uint16, (0, 65535)),
    "int16": (TP.INT16, np.int16, (-32768, 32767)),
    "int32": (TP.INT32, np.int32, (-(2**31), 2**31 - 1)),
    "int4": (TP.INT4, np.int8, (-8, 7)),
    "uint4": (TP.UINT4, np.uint8, (0, 15)),
    # float8 tensors are carried as their raw uint8 bit patterns
    "float8e4m3fn": (TP.FLOAT8E4M3FN, np.uint8, None),
    "float8e5m2": (TP.FLOAT8E5M2, np.uint8, None),
}
INT_TYPES = {k for k, v in DTYPES.items() if v[2] is not None}
FLOAT8_TYPES = {"float8e4m3fn", "float8e5m2"}
# types numpy/ORT python bindings cannot feed or fetch directly
OPAQUE_TYPES = {"int4", "uint4"}
ELEM_TO_NAME = {v[0]: k for k, v in DTYPES.items()}

OPSET = 21
MS_OPSET = 1


def qrange(dtype: str) -> tuple[int, int]:
    r = DTYPES[dtype][2]
    if r is None:
        raise ValueError(f"{dtype} is not an integer type")
    return r


@dataclass
class TensorSpec:
    name: str
    dtype: str
    data: np.ndarray  # carrier dtype (see DTYPES)
    dims: list[str | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=DTYPES[self.dtype][1])
        if not self.dims:
            self.dims = [None] * self.data.ndim

    def to_json(self) -> dict:
        if self.dtype == "float32":
            flat = [float(v) for v in self.data.astype(np.float64).ravel()]
        else:
            flat = [int(v) for v in self.data.ravel()]
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.data.shape),
                "data": flat, "dims": self.dims}

    @staticmethod
    def from_json(d: dict) -> "TensorSpec":
        arr = np.array(d["data"], dtype=np.float64 if d["dtype"] == "float32" else np.int64)
        arr = arr.reshape(d["shape"]).astype(DTYPES[d["dtype"]][1])
        return TensorSpec(d["name"], d["dtype"], arr, list(d.get("dims") or []))

    def to_onnx(self) -> onnx.TensorProto:
        elem = DTYPES[self.dtype][0]
        shape = list(self.data.shape)
        if self.dtype in FLOAT8_TYPES:
            return helper.make_tensor(self.name, elem, shape, self.data.astype(np.uint8).tobytes(), raw=True)
        if self.dtype in OPAQUE_TYPES:
            return helper.make_tensor(self.name, elem, shape, [int(v) for v in self.data.ravel()])
        return onnx.numpy_helper.from_array(self.data, self.name)


@dataclass
class NodeSpec:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    domain: str = ""

    def to_json(self) -> dict:
        return {"op": self.op, "inputs": self.inputs, "outputs": self.outputs,
                "attrs": self.attrs, "domain": self.domain}

    @staticmethod
    def from_json(d: dict) -> "NodeSpec":
        return NodeSpec(d["op"], list(d["inputs"]), list(d["outputs"]), dict(d.get("attrs", {})), d.get("domain", ""))


@dataclass
class Case:
    nodes: list[NodeSpec]
    inputs: list[TensorSpec]  # graph inputs, with the values we feed
    inits: list[TensorSpec]  # initializers (constants)
    outputs: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    opset: int = OPSET

    # ------------------------------------------------------------------ utils
    def copy(self) -> "Case":
        return copy.deepcopy(self)

    def tensor(self, name: str) -> TensorSpec | None:
        for t in self.inputs + self.inits:
            if t.name == name:
                return t
        return None

    def producer(self, name: str) -> NodeSpec | None:
        for n in self.nodes:
            if name in n.outputs:
                return n
        return None

    def op_types(self) -> list[str]:
        return sorted({n.op for n in self.nodes})

    def uses_ms(self) -> bool:
        return any(n.domain == "com.microsoft" for n in self.nodes)

    def size(self) -> int:
        """Rough size measure used by the minimizer (smaller is better)."""
        return 1000 * len(self.nodes) + sum(t.data.size for t in self.inputs + self.inits)

    # --------------------------------------------------------------- (de)ser
    def to_json(self) -> dict:
        return {"opset": self.opset, "nodes": [n.to_json() for n in self.nodes],
                "inputs": [t.to_json() for t in self.inputs],
                "inits": [t.to_json() for t in self.inits],
                "outputs": list(self.outputs), "meta": self.meta}

    @staticmethod
    def from_json(d: dict) -> "Case":
        return Case([NodeSpec.from_json(n) for n in d["nodes"]],
                    [TensorSpec.from_json(t) for t in d["inputs"]],
                    [TensorSpec.from_json(t) for t in d["inits"]],
                    list(d["outputs"]), dict(d.get("meta", {})), int(d.get("opset", OPSET)))

    def dumps(self) -> str:
        return json.dumps(self.to_json())

    # ------------------------------------------------------------------ onnx
    def to_model(self) -> onnx.ModelProto:
        nodes = []
        for i, n in enumerate(self.nodes):
            nodes.append(helper.make_node(n.op, n.inputs, n.outputs, name=f"n{i}_{n.op}",
                                          domain=n.domain or None, **n.attrs))
        g_inputs = [helper.make_tensor_value_info(t.name, DTYPES[t.dtype][0], list(t.data.shape))
                    for t in self.inputs]
        types = self.dtypes()
        shapes = self.output_shapes()
        g_outputs = [helper.make_tensor_value_info(o, DTYPES[types[o]][0], shapes.get(o)) if o in types
                     else helper.make_empty_tensor_value_info(o) for o in self.outputs]
        graph = helper.make_graph(nodes, "qfuzz", g_inputs, g_outputs,
                                  initializer=[t.to_onnx() for t in self.inits])
        opsets = [helper.make_opsetid("", self.opset)]
        if self.uses_ms():
            opsets.append(helper.make_opsetid("com.microsoft", MS_OPSET))
        model = helper.make_model(graph, opset_imports=opsets, producer_name="qfuzz")
        model.ir_version = 10
        return model

    def output_shapes(self) -> dict[str, list]:
        """Output shapes from the reference interpreter (ORT-independent)."""
        from . import reference  # local import: reference depends on this module

        try:
            env = reference.evaluate(self)
            return {o: list(np.shape(env[o].value)) for o in self.outputs}
        except Exception:  # noqa: BLE001 - invalid cases fall back to unknown shape
            return {}

    def dtypes(self) -> dict[str, str]:
        """Static dtype propagation (needed for output value_info, incl. contrib ops)."""
        t = {x.name: x.dtype for x in self.inputs + self.inits}
        for n in self.nodes:
            ins = [t.get(i) for i in n.inputs]
            if n.op == "QuantizeLinear":
                if len(ins) > 2 and ins[2]:
                    o = [ins[2]]
                else:
                    o = [ELEM_TO_NAME.get(n.attrs.get("output_dtype", TP.UINT8), "uint8")]
            elif n.op in ("DequantizeLinear", "MatMul", "Conv", "Add", "Mul"):
                o = ["float32"]
            elif n.op in ("MatMulInteger", "ConvInteger"):
                o = ["int32"]
            elif n.op in ("QLinearMatMul", "QLinearConv"):
                o = [ins[7]]
            elif n.op in ("QLinearAdd", "QLinearMul"):
                o = [ins[7] if len(ins) > 7 and ins[7] else ins[0]]
            elif n.op == "DynamicQuantizeLinear":
                o = ["uint8", "float32", "uint8"]
            elif n.op == "Cast":
                o = [ELEM_TO_NAME[n.attrs["to"]]]
            else:  # Relu, Identity, ...
                o = [ins[0]]
            for name, dt in zip(n.outputs, o):
                if name and dt:
                    t[name] = dt
        return t

    def feeds(self) -> dict[str, np.ndarray]:
        return {t.name: t.data for t in self.inputs}

    def prune(self) -> "Case":
        """Drop nodes / tensors that do not contribute to any output."""
        needed = set(self.outputs)
        keep: list[NodeSpec] = []
        for n in reversed(self.nodes):
            if any(o in needed for o in n.outputs):
                keep.append(n)
                needed.update(i for i in n.inputs if i)
        keep.reverse()
        c = self.copy()
        c.nodes = keep
        c.inputs = [t for t in c.inputs if t.name in needed]
        c.inits = [t for t in c.inits if t.name in needed]
        return c
