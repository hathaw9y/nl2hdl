from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

from .graph import DenseLayer, ModelGraph


SUPPORTED_OPS = {"Gemm", "MatMul", "Add", "Relu", "Flatten", "Reshape", "Identity"}


class UnsupportedModelError(RuntimeError):
    def __init__(self, message: str, unsupported_ops: list[str] | None = None) -> None:
        super().__init__(message)
        self.unsupported_ops = unsupported_ops or []


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}


def _attr(node: onnx.NodeProto, name: str, default: int | float) -> int | float:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return int(attr.i)
            if attr.type == onnx.AttributeProto.FLOAT:
                return float(attr.f)
    return default


def _input_size(model: onnx.ModelProto, initializers: dict[str, np.ndarray]) -> tuple[str, int]:
    for value in model.graph.input:
        if value.name in initializers:
            continue
        dims = value.type.tensor_type.shape.dim
        shape = [int(dim.dim_value) for dim in dims if dim.dim_value > 0]
        if not shape or shape[0] != 1:
            raise UnsupportedModelError("v1 requires a fixed batch size of 1")
        return value.name, int(np.prod(shape[1:]))
    raise UnsupportedModelError("model has no non-initializer input")


def _as_dense_weight(weight: np.ndarray, input_size: int, transposed: bool) -> np.ndarray:
    if weight.ndim != 2:
        raise UnsupportedModelError("dense weights must be rank-2")
    candidate = weight.T if transposed else weight
    if candidate.shape[1] == input_size:
        return candidate.astype(np.float32)
    if candidate.shape[0] == input_size:
        return candidate.T.astype(np.float32)
    raise UnsupportedModelError(
        f"dense weight shape {tuple(weight.shape)} is incompatible with input size {input_size}"
    )


def parse_onnx_graph(onnx_path: Path, name: str) -> ModelGraph:
    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializers = _initializer_map(model)
    unsupported = sorted({node.op_type for node in model.graph.node if node.op_type not in SUPPORTED_OPS})
    if unsupported:
        raise UnsupportedModelError(
            "unsupported ops in model graph: " + ", ".join(unsupported),
            unsupported_ops=unsupported,
        )

    input_name, current_size = _input_size(model, initializers)
    current_name = input_name
    layers: list[DenseLayer] = []
    aliases: dict[str, str] = {}

    def resolve(value: str) -> str:
        while value in aliases:
            value = aliases[value]
        return value

    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        if op in {"Flatten", "Reshape", "Identity"}:
            aliases[node.output[0]] = resolve(node.input[0])
            current_name = node.output[0] if resolve(node.input[0]) == resolve(current_name) else current_name
            continue

        if op in {"Gemm", "MatMul"}:
            inputs = list(node.input)
            activation_input = resolve(inputs[0])
            if activation_input != resolve(current_name):
                raise UnsupportedModelError(f"{op} node {node.name or idx} is not in a simple sequential graph")
            weight_name = inputs[1]
            if weight_name not in initializers:
                raise UnsupportedModelError(f"{op} node {node.name or idx} has non-constant weights")
            trans_b = bool(_attr(node, "transB", 0)) if op == "Gemm" else False
            weights = _as_dense_weight(initializers[weight_name], current_size, trans_b)
            bias = np.zeros((weights.shape[0],), dtype=np.float32)
            if op == "Gemm" and len(inputs) >= 3 and inputs[2] in initializers:
                bias = initializers[inputs[2]].reshape(-1).astype(np.float32)
            if bias.shape[0] != weights.shape[0]:
                raise UnsupportedModelError(f"bias size is incompatible in node {node.name or idx}")
            layer = DenseLayer(
                name=f"dense_{len(layers)}",
                input_name=current_name,
                output_name=node.output[0],
                weights=weights,
                bias=bias,
            )
            layers.append(layer)
            current_name = node.output[0]
            current_size = layer.output_size
            continue

        if op == "Add":
            if not layers:
                raise UnsupportedModelError("Add before first dense layer is not supported")
            a_name = resolve(node.input[0])
            b_name = resolve(node.input[1])
            bias_name = None
            if a_name == resolve(current_name) and b_name in initializers:
                bias_name = b_name
            elif b_name == resolve(current_name) and a_name in initializers:
                bias_name = a_name
            if bias_name is None:
                raise UnsupportedModelError("v1 supports Add only as dense bias")
            bias = initializers[bias_name].reshape(-1).astype(np.float32)
            if bias.shape[0] != layers[-1].output_size:
                raise UnsupportedModelError("Add bias size is incompatible with previous dense layer")
            layers[-1].bias = layers[-1].bias + bias
            layers[-1].output_name = node.output[0]
            current_name = node.output[0]
            continue

        if op == "Relu":
            if not layers or resolve(node.input[0]) != resolve(current_name):
                raise UnsupportedModelError("Relu must follow a supported dense layer")
            layers[-1].activation = "relu"
            layers[-1].output_name = node.output[0]
            current_name = node.output[0]
            continue

    if not layers:
        raise UnsupportedModelError("model contains no supported dense layers")
    return ModelGraph(
        name=name.replace("/", "_").replace(":", "_"),
        input_name=input_name,
        input_size=layers[0].input_size,
        output_name=current_name,
        layers=layers,
    )
