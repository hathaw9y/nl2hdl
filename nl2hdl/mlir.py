from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re


GRAPH_SUPPORTED_OPS = {
    "Gemm",
    "MatMul",
    "Add",
    "Relu",
    "Flatten",
    "Reshape",
    "Identity",
    "RMSNorm",
    "RoPE",
    "Softmax",
    "SiLU",
    "Mul",
    "AttentionControl",
}
IGNORED_ONNX_OPS = {"Constant", "EntryPoint"}
GEMM_OPS = {"Gemm", "MatMul"}


@dataclass(frozen=True)
class MlirTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MlirOp:
    result: str | None
    op_type: str
    operands: tuple[str, ...]
    result_shape: tuple[int, ...] | None
    result_dtype: str | None
    node_name: str | None


@dataclass(frozen=True)
class MlirAnalysis:
    entry: str | None
    inputs: tuple[MlirTensor, ...]
    outputs: tuple[MlirTensor, ...]
    ops: tuple[MlirOp, ...]
    constants: int
    supported_ops: tuple[str, ...]
    unsupported_ops: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "inputs": [
                {"name": item.name, "shape": list(item.shape), "dtype": item.dtype}
                for item in self.inputs
            ],
            "outputs": [
                {"name": item.name, "shape": list(item.shape), "dtype": item.dtype}
                for item in self.outputs
            ],
            "ops": [
                {
                    "result": op.result,
                    "op_type": op.op_type,
                    "operands": list(op.operands),
                    "result_shape": list(op.result_shape) if op.result_shape is not None else None,
                    "result_dtype": op.result_dtype,
                    "node_name": op.node_name,
                }
                for op in self.ops
            ],
            "constants": self.constants,
            "supported_ops": list(self.supported_ops),
            "unsupported_ops": list(self.unsupported_ops),
            "op_partition": {
                "gemm": [op.op_type for op in self.ops if op.op_type in GEMM_OPS],
                "non_gemm": [op.op_type for op in self.ops if op.op_type not in GEMM_OPS],
            },
        }


def _parse_tensor(text: str) -> tuple[tuple[int, ...], str] | None:
    match = re.search(r"tensor<([^>]+)>", text)
    if not match:
        return None
    parts = match.group(1).split("x")
    if not parts:
        return None
    dtype = parts[-1]
    shape: list[int] = []
    for part in parts[:-1]:
        if part == "?":
            return None
        try:
            shape.append(int(part))
        except ValueError:
            return None
    return tuple(shape), dtype


def _parse_entry(line: str) -> tuple[str | None, tuple[MlirTensor, ...], tuple[MlirTensor, ...]]:
    match = re.search(r"func\.func\s+@([^(]+)\((.*?)\)\s*->\s*\((.*?)\)", line)
    if not match:
        return None, tuple(), tuple()
    entry = match.group(1)
    inputs: list[MlirTensor] = []
    for arg_match in re.finditer(r"(%arg\d+):\s*(tensor<[^>]+>)", match.group(2)):
        parsed = _parse_tensor(arg_match.group(2))
        if parsed:
            shape, dtype = parsed
            inputs.append(MlirTensor(arg_match.group(1), shape, dtype))
    outputs: list[MlirTensor] = []
    parsed_output = _parse_tensor(match.group(3))
    if parsed_output:
        shape, dtype = parsed_output
        outputs.append(MlirTensor("return", shape, dtype))
    return entry, tuple(inputs), tuple(outputs)


def analyze_mlir(mlir_path: Path) -> MlirAnalysis:
    text = mlir_path.read_text(encoding="utf-8", errors="ignore")
    entry = None
    inputs: tuple[MlirTensor, ...] = tuple()
    outputs: tuple[MlirTensor, ...] = tuple()
    ops: list[MlirOp] = []
    constants = 0

    for line in text.splitlines():
        stripped = line.strip()
        if "func.func" in stripped and entry is None:
            entry, inputs, outputs = _parse_entry(stripped)
            continue

        if "onnx.Constant" in stripped:
            constants += 1
            continue

        quoted_match = re.search(
            r'(?:(%\w+)\s*=\s*)?"(?:onnx|llm)\.([A-Za-z0-9_]+)"\(([^)]*)\).*?->\s*(tensor<[^>]+>)',
            stripped,
        )
        bare_match = None
        if not quoted_match:
            bare_match = re.search(r'(?:(%\w+)\s*=\s*)?(?:onnx|llm)\.([A-Za-z0-9_]+)\b.*?->\s*(tensor<[^>]+>)', stripped)
        if quoted_match:
            result = quoted_match.group(1)
            op_type = quoted_match.group(2)
            operands_raw = quoted_match.group(3)
            result_tensor = quoted_match.group(4)
        elif bare_match:
            result = bare_match.group(1)
            op_type = bare_match.group(2)
            operands_raw = ""
            result_tensor = bare_match.group(3)
        else:
            continue
        operands = tuple(item.strip() for item in operands_raw.split(",") if item.strip().startswith("%"))
        parsed_result = _parse_tensor(result_tensor)
        node_name_match = re.search(r'onnx_node_name\s*=\s*"([^"]+)"', stripped)
        if not node_name_match:
            node_name_match = re.search(r'loc\("([^"]+)"\)', stripped)
        node_name = node_name_match.group(1) if node_name_match else None
        ops.append(
            MlirOp(
                result=result,
                op_type=op_type,
                operands=operands,
                result_shape=parsed_result[0] if parsed_result else None,
                result_dtype=parsed_result[1] if parsed_result else None,
                node_name=node_name,
            )
        )

    graph_ops = [op.op_type for op in ops if op.op_type not in IGNORED_ONNX_OPS]
    supported = sorted(set(op for op in graph_ops if op in GRAPH_SUPPORTED_OPS))
    unsupported = sorted(set(op for op in graph_ops if op not in GRAPH_SUPPORTED_OPS))
    return MlirAnalysis(
        entry=entry,
        inputs=inputs,
        outputs=outputs,
        ops=tuple(ops),
        constants=constants,
        supported_ops=tuple(supported),
        unsupported_ops=tuple(unsupported),
    )


def save_mlir_analysis(analysis: MlirAnalysis, path: Path) -> None:
    path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
