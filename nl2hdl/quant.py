from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .graph import DenseLayer, ModelGraph


@dataclass
class QuantizedLayer:
    name: str
    input_size: int
    output_size: int
    weights_i8: np.ndarray
    bias_i32: np.ndarray
    requant_mult: int
    requant_shift: int
    activation: str
    input_scale: float
    weight_scale: float
    output_scale: float


@dataclass
class QuantizedModel:
    input_size: int
    output_size: int
    input_scale: float
    output_scale: float
    input_i8: np.ndarray
    expected_i8: np.ndarray
    layers: list[QuantizedLayer]

    def to_report(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_scale": self.input_scale,
            "output_scale": self.output_scale,
            "input_i8": self.input_i8.reshape(-1).astype(int).tolist(),
            "expected_i8": self.expected_i8.reshape(-1).astype(int).tolist(),
            "layers": [
                {
                    "name": layer.name,
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "activation": layer.activation,
                    "input_scale": layer.input_scale,
                    "weight_scale": layer.weight_scale,
                    "output_scale": layer.output_scale,
                    "requant_mult": layer.requant_mult,
                    "requant_shift": layer.requant_shift,
                }
                for layer in self.layers
            ],
        }


def _scale(values: np.ndarray) -> float:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    return max(max_abs / 127.0, 1.0 / 127.0)


def _q_i8(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(values / scale), -128, 127).astype(np.int8)


def _clip_i8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -128, 127).astype(np.int8)


def _run_float(layer: DenseLayer, x: np.ndarray) -> np.ndarray:
    y = x.reshape(-1).astype(np.float32) @ layer.weights.T + layer.bias
    if layer.activation == "relu":
        y = np.maximum(y, 0)
    return y.astype(np.float32)


def quantize_graph(graph: ModelGraph, sample_input: np.ndarray, requant_shift: int = 16) -> QuantizedModel:
    x_float = sample_input.reshape(-1).astype(np.float32)
    input_scale = _scale(x_float)
    x_i8 = _q_i8(x_float, input_scale)
    current_scale = input_scale
    current_i8 = x_i8.astype(np.int32)
    current_float = x_float
    q_layers: list[QuantizedLayer] = []

    for layer in graph.layers:
        y_float = _run_float(layer, current_float)
        output_scale = _scale(y_float)
        weight_scale = _scale(layer.weights)
        weights_i8 = _q_i8(layer.weights, weight_scale)
        bias_i32 = np.rint(layer.bias / (current_scale * weight_scale)).astype(np.int32)
        real_requant = (current_scale * weight_scale) / output_scale
        requant_mult = max(1, int(round(real_requant * (1 << requant_shift))))

        acc = current_i8.astype(np.int32) @ weights_i8.astype(np.int32).T + bias_i32
        y_i32 = (acc.astype(np.int64) * requant_mult) >> requant_shift
        if layer.activation == "relu":
            y_i32 = np.maximum(y_i32, 0)
        y_i8 = _clip_i8(y_i32)

        q_layers.append(
            QuantizedLayer(
                name=layer.name,
                input_size=layer.input_size,
                output_size=layer.output_size,
                weights_i8=weights_i8,
                bias_i32=bias_i32,
                requant_mult=requant_mult,
                requant_shift=requant_shift,
                activation=layer.activation,
                input_scale=current_scale,
                weight_scale=weight_scale,
                output_scale=output_scale,
            )
        )
        current_scale = output_scale
        current_i8 = y_i8.astype(np.int32)
        current_float = y_float

    return QuantizedModel(
        input_size=graph.input_size,
        output_size=graph.output_size,
        input_scale=input_scale,
        output_scale=current_scale,
        input_i8=x_i8.reshape(-1),
        expected_i8=current_i8.astype(np.int8).reshape(-1),
        layers=q_layers,
    )


def save_quant_report(qmodel: QuantizedModel, path: Path) -> None:
    path.write_text(json.dumps(qmodel.to_report(), indent=2), encoding="utf-8")
