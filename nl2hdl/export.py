from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import torch

from .config import AgentConfig


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(
                torch.tensor(
                    [
                        [0.25, -0.50, 0.75, 0.10],
                        [-0.30, 0.20, 0.40, -0.60],
                        [0.90, 0.10, -0.20, 0.30],
                        [-0.70, 0.80, 0.05, 0.20],
                        [0.15, -0.35, 0.55, 0.45],
                    ],
                    dtype=torch.float32,
                )
            )
            self.net[0].bias.copy_(torch.tensor([0.10, -0.20, 0.05, 0.30, -0.10]))
            self.net[2].weight.copy_(
                torch.tensor(
                    [
                        [0.40, -0.10, 0.25, 0.60, -0.30],
                        [-0.50, 0.20, 0.70, -0.40, 0.10],
                        [0.30, 0.80, -0.20, 0.15, 0.50],
                    ],
                    dtype=torch.float32,
                )
            )
            self.net[2].bias.copy_(torch.tensor([0.05, -0.15, 0.20]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def deterministic_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.5, size=shape).astype(np.float32)


def export_model_to_onnx(model_name: str, config: AgentConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    dummy_np = deterministic_input(config.model.input_shape, config.model.dummy_input_seed)
    dummy = torch.from_numpy(dummy_np)
    np.save(out_dir / "dummy_input.npy", dummy_np)

    if model_name == "builtin:tiny_mlp":
        model = TinyMlp().eval()
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
        return onnx_path

    from transformers import AutoConfig, AutoModel

    hf_config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=hf_config).eval()
    if getattr(hf_config, "model_type", None) in {"bert", "roberta", "distilbert", "gpt2"}:
        seq_len = config.model.sequence_length
        input_ids = torch.zeros((1, seq_len), dtype=torch.long)
        torch.onnx.export(
            model,
            (input_ids,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
    return onnx_path


def emit_onnx_mlir(onnx_path: Path, out_dir: Path) -> Path:
    mlir_path = out_dir / "model_graph.mlir"
    onnx_mlir = (
        os.environ.get("ONNX_MLIR")
        or shutil.which("onnx-mlir")
        or str(Path("~/onnx/onnx-mlir/build/Release/bin/onnx-mlir").expanduser())
    )
    result = subprocess.run(
        [onnx_mlir, "--EmitONNXIR", str(onnx_path), "-o", str(out_dir / "model_graph")],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"onnx-mlir failed while emitting ONNX MLIR:\n{result.stdout}")
    produced = out_dir / "model_graph.onnx.mlir"
    if not produced.exists():
        raise RuntimeError(f"onnx-mlir did not produce expected MLIR file: {produced}")
    produced.rename(mlir_path)
    return mlir_path
