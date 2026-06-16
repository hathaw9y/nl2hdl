from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import numpy as np


@dataclass
class DenseLayer:
    name: str
    input_name: str
    output_name: str
    weights: np.ndarray
    bias: np.ndarray
    activation: str = "linear"

    @property
    def input_size(self) -> int:
        return int(self.weights.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.weights.shape[0])


@dataclass
class ModelGraph:
    name: str
    input_name: str
    input_size: int
    output_name: str
    layers: list[DenseLayer]

    @property
    def output_size(self) -> int:
        return self.layers[-1].output_size if self.layers else self.input_size

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_name": self.input_name,
            "input_size": self.input_size,
            "output_name": self.output_name,
            "output_size": self.output_size,
            "layers": [
                {
                    "name": layer.name,
                    "input_name": layer.input_name,
                    "output_name": layer.output_name,
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "activation": layer.activation,
                }
                for layer in self.layers
            ],
        }


def save_graph_summary(graph: ModelGraph, path: Path) -> None:
    path.write_text(json.dumps(graph.to_summary(), indent=2), encoding="utf-8")


def layer_to_dict(layer: DenseLayer) -> dict[str, Any]:
    data = asdict(layer)
    data["weights"] = layer.weights.tolist()
    data["bias"] = layer.bias.tolist()
    return data
