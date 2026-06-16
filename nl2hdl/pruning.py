from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .config import OptimizationConfig
from .graph import DenseLayer, ModelGraph


def apply_pruning(graph: ModelGraph, optimization: OptimizationConfig) -> tuple[ModelGraph, dict[str, Any]]:
    if optimization.pruning == "none":
        report = {
            "method": "none",
            "threshold": optimization.pruning_threshold,
            "total_weights": int(sum(layer.weights.size for layer in graph.layers)),
            "zeroed_weights": 0,
            "sparsity": 0.0,
            "hardware_note": "dense RTL is emitted without zero-skip control",
        }
        return graph, report

    if optimization.pruning != "magnitude_unstructured":
        raise ValueError(f"unsupported pruning method: {optimization.pruning}")

    threshold = float(optimization.pruning_threshold)
    zeroed_total = 0
    total = 0
    pruned_layers: list[DenseLayer] = []
    layer_reports: list[dict[str, Any]] = []
    for layer in graph.layers:
        weights = layer.weights.copy()
        mask = np.abs(weights) <= threshold
        zeroed = int(np.count_nonzero(mask))
        weights[mask] = 0.0
        total += int(weights.size)
        zeroed_total += zeroed
        layer_reports.append(
            {
                "name": layer.name,
                "total_weights": int(weights.size),
                "zeroed_weights": zeroed,
                "sparsity": float(zeroed / weights.size) if weights.size else 0.0,
            }
        )
        pruned_layers.append(
            DenseLayer(
                name=layer.name,
                input_name=layer.input_name,
                output_name=layer.output_name,
                weights=weights,
                bias=layer.bias.copy(),
                activation=layer.activation,
            )
        )
    pruned_graph = ModelGraph(
        name=graph.name,
        input_name=graph.input_name,
        input_size=graph.input_size,
        output_name=graph.output_name,
        layers=pruned_layers,
    )
    report = {
        "method": optimization.pruning,
        "threshold": threshold,
        "total_weights": total,
        "zeroed_weights": zeroed_total,
        "sparsity": float(zeroed_total / total) if total else 0.0,
        "layers": layer_reports,
        "hardware_note": "weights are statically zeroed before dense direct-RTL generation; no runtime sparse index format is emitted",
    }
    return pruned_graph, report


def save_pruning_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
