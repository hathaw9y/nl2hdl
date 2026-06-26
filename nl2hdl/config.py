from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass(frozen=True)
class ModelConfig:
    input_shape: tuple[int, ...] = (1, 4)
    dummy_input_seed: int = 7
    sequence_length: int = 8
    gptq_checkpoint: str | None = None
    mlir_graph: str | None = None
    model_structure_source: str = "mlir"


@dataclass(frozen=True)
class HardwareConfig:
    fpga_part: str = "xc7z020clg400-1"
    target_clock_mhz: int = 50
    max_dsp: int = 90
    max_bram: int = 50
    max_lut: int = 20000
    max_ff: int | None = None
    max_uram: int | None = None
    max_io: int | None = None
    memory_data_width: int = 32
    device_logic_cells: int | None = None
    device_lut: int | None = None
    device_ff: int | None = None
    device_dsp: int | None = None
    device_bram_36k: int | None = None
    device_uram: int | None = None
    device_io: int | None = None
    device_distributed_ram_mb: float | None = None
    device_bram_mb: float | None = None
    device_uram_mb: float | None = None
    device_ps_gtr: int | None = None
    device_gth: int | None = None
    resource_reference: str | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    quantization: str = "int8_static"
    pruning: str = "none"
    pruning_threshold: float = 0.0
    optimization_brief: str | None = None
    optimization_candidates: list[Any] = field(default_factory=list)
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignConfig:
    style: str = "layer_fsm"
    compute_style: str = "scalar_fsm"
    execution_style: str = "layer_by_layer"
    memory_style: str = "onchip_weight_storage"
    control_style: str = "layer_fsm"
    pe_count: int = 4
    activation_buffer: str = "register_ping_pong"
    weight_storage: str = "case_rom"
    architecture_brief: str | None = None
    design_candidates: list[Any] = field(default_factory=list)
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationConfig:
    enable_verilator: bool = True
    enable_vivado_synth: bool = True
    tolerance_lsb: int = 1
    vivado_timeout_sec: int = 300


@dataclass(frozen=True)
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    retry_count: int = 1


def _merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of positive integers")
    shape = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{field_name} must contain only positive dimensions")
    if shape[0] != 1:
        raise ValueError("v1 requires batch size 1")
    return shape


def _split_extra_options(section: dict[str, Any], config_cls: type, extra_field: str) -> dict[str, Any]:
    known_fields = set(config_cls.__dataclass_fields__)
    normalized = dict(section)
    extra = normalized.get(extra_field, {})
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError(f"{extra_field} must be a mapping/object when provided")
    extra = dict(extra)
    for key in list(normalized):
        if key not in known_fields:
            extra[key] = normalized.pop(key)
    normalized[extra_field] = extra
    return normalized


def _require_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def default_config_dict() -> dict[str, Any]:
    return {
        "model": {
            "input_shape": list(ModelConfig().input_shape),
            "dummy_input_seed": ModelConfig().dummy_input_seed,
            "sequence_length": ModelConfig().sequence_length,
            "gptq_checkpoint": ModelConfig().gptq_checkpoint,
            "mlir_graph": ModelConfig().mlir_graph,
            "model_structure_source": ModelConfig().model_structure_source,
        },
        "hardware": HardwareConfig().__dict__,
        "optimization": OptimizationConfig().__dict__,
        "design": DesignConfig().__dict__,
        "verification": VerificationConfig().__dict__,
        "retry_count": AgentConfig().retry_count,
    }


def load_config(path: str | Path | None) -> AgentConfig:
    raw: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            if config_path.suffix.lower() == ".json":
                loaded = json.load(handle)
            else:
                loaded = yaml.safe_load(handle)
        raw = loaded or {}
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a mapping/object at the top level")

    data = _merge(default_config_dict(), raw)
    model_raw = data["model"]
    model = ModelConfig(
        input_shape=_as_tuple(model_raw["input_shape"], "model.input_shape"),
        dummy_input_seed=int(model_raw["dummy_input_seed"]),
        sequence_length=int(model_raw["sequence_length"]),
        gptq_checkpoint=(
            str(model_raw["gptq_checkpoint"]).strip() if model_raw.get("gptq_checkpoint") is not None else None
        ),
        mlir_graph=(str(model_raw["mlir_graph"]).strip() if model_raw.get("mlir_graph") is not None else None),
        model_structure_source=str(model_raw.get("model_structure_source", ModelConfig().model_structure_source)).strip(),
    )
    hardware = HardwareConfig(**data["hardware"])
    optimization_raw = _split_extra_options(data["optimization"], OptimizationConfig, "extra_options")
    optimization = OptimizationConfig(**optimization_raw)
    raw_design = raw.get("design", {}) if isinstance(raw.get("design", {}), dict) else {}
    design_raw = _split_extra_options(data["design"], DesignConfig, "extra_options")
    style = str(design_raw.get("style", DesignConfig().style))
    explicit_execution_style = "execution_style" in raw_design
    if not explicit_execution_style and style == "llm_decoder_streaming":
        design_raw["execution_style"] = "llm_decoder_streaming"
    if not explicit_execution_style and style == "layer_fsm":
        design_raw["execution_style"] = "layer_by_layer"
    if "control_style" not in design_raw and style == "layer_fsm":
        design_raw["control_style"] = "layer_fsm"
    if "style" not in design_raw and explicit_execution_style:
        design_raw["style"] = design_raw["execution_style"]
    design = DesignConfig(**design_raw)
    verification = VerificationConfig(**data["verification"])
    cfg = AgentConfig(
        model=model,
        hardware=hardware,
        optimization=optimization,
        design=design,
        verification=verification,
        retry_count=int(data["retry_count"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(config: AgentConfig) -> None:
    _require_non_empty_string(config.optimization.quantization, "optimization.quantization")
    _require_non_empty_string(config.optimization.pruning, "optimization.pruning")
    if config.model.gptq_checkpoint is not None and not config.model.gptq_checkpoint:
        raise ValueError("model.gptq_checkpoint must be a non-empty path or Hugging Face repo id when provided")
    if config.model.mlir_graph is not None and not config.model.mlir_graph:
        raise ValueError("model.mlir_graph must be a non-empty path when provided")
    if config.model.model_structure_source not in {"mlir", "hf_config"}:
        raise ValueError("model.model_structure_source must be one of: mlir, hf_config")
    if config.optimization.pruning_threshold < 0.0:
        raise ValueError("optimization.pruning_threshold must be non-negative")
    if not isinstance(config.optimization.optimization_candidates, list):
        raise ValueError("optimization.optimization_candidates must be a list when provided")
    if not isinstance(config.optimization.extra_options, dict):
        raise ValueError("optimization.extra_options must be a mapping/object")
    _require_non_empty_string(config.design.style, "design.style")
    _require_non_empty_string(config.design.compute_style, "design.compute_style")
    _require_non_empty_string(config.design.execution_style, "design.execution_style")
    _require_non_empty_string(config.design.memory_style, "design.memory_style")
    _require_non_empty_string(config.design.control_style, "design.control_style")
    if not isinstance(config.design.design_candidates, list):
        raise ValueError("design.design_candidates must be a list when provided")
    if not isinstance(config.design.extra_options, dict):
        raise ValueError("design.extra_options must be a mapping/object")
    if config.design.pe_count <= 0:
        raise ValueError("design.pe_count must be positive")
    if config.hardware.memory_data_width not in (8, 16, 32, 64, 128):
        raise ValueError("hardware.memory_data_width must be one of 8, 16, 32, 64, 128")
    if config.hardware.target_clock_mhz <= 0:
        raise ValueError("hardware.target_clock_mhz must be positive")
    if config.hardware.max_dsp < 0 or config.hardware.max_bram < 0 or config.hardware.max_lut < 0:
        raise ValueError("hardware max resource budgets must be non-negative")
    optional_non_negative_fields = {
        "max_ff": config.hardware.max_ff,
        "max_uram": config.hardware.max_uram,
        "max_io": config.hardware.max_io,
        "device_logic_cells": config.hardware.device_logic_cells,
        "device_lut": config.hardware.device_lut,
        "device_ff": config.hardware.device_ff,
        "device_dsp": config.hardware.device_dsp,
        "device_bram_36k": config.hardware.device_bram_36k,
        "device_uram": config.hardware.device_uram,
        "device_io": config.hardware.device_io,
        "device_distributed_ram_mb": config.hardware.device_distributed_ram_mb,
        "device_bram_mb": config.hardware.device_bram_mb,
        "device_uram_mb": config.hardware.device_uram_mb,
        "device_ps_gtr": config.hardware.device_ps_gtr,
        "device_gth": config.hardware.device_gth,
    }
    for field_name, value in optional_non_negative_fields.items():
        if value is not None and value < 0:
            raise ValueError(f"hardware.{field_name} must be non-negative")
