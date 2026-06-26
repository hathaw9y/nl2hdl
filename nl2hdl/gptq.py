from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import fnmatch
import hashlib
import json
import os
import re
import struct

import numpy as np


GPTQ_CHECKPOINT_ALLOW_PATTERNS = [
    "config.json",
    "quantize_config.json",
    "quant_config.json",
    "gptq_config.json",
    "quantization_config.json",
    "*.safetensors.index.json",
    "*.bin.index.json",
    "*.safetensors",
]


@dataclass(frozen=True)
class GptqProjection:
    name: str
    rows: int
    cols: int
    group_size: int
    qweight: bytes
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    signed: bool = True


def pack_int4(values: np.ndarray, signed: bool = True) -> bytes:
    flat = values.reshape(-1).astype(np.int32)
    encoded: list[int] = []
    for value in flat:
        if signed:
            if value < -8 or value > 7:
                raise ValueError("signed int4 values must be in [-8, 7]")
            encoded.append(value & 0xF)
        else:
            if value < 0 or value > 15:
                raise ValueError("unsigned int4 values must be in [0, 15]")
            encoded.append(int(value))
    packed = bytearray()
    for idx in range(0, len(encoded), 2):
        lo = encoded[idx]
        hi = encoded[idx + 1] if idx + 1 < len(encoded) else 0
        packed.append((hi << 4) | lo)
    return bytes(packed)


def unpack_int4(data: bytes, count: int, signed: bool = True) -> np.ndarray:
    values: list[int] = []
    for byte in data:
        values.append(byte & 0xF)
        values.append((byte >> 4) & 0xF)
    values = values[:count]
    if signed:
        values = [value - 16 if value >= 8 else value for value in values]
    return np.array(values, dtype=np.int8)


def dequantize_int4(values: np.ndarray, scales: np.ndarray, zeros: np.ndarray, group_size: int) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float32)
    out = np.zeros_like(flat, dtype=np.float32)
    for idx, value in enumerate(flat):
        group_idx = min(idx // group_size, len(scales) - 1)
        out[idx] = (value - zeros[group_idx]) * scales[group_idx]
    return out.reshape(values.shape)


def synthetic_projection() -> GptqProjection:
    weights = np.array(
        [
            [1, -2, 3, -4],
            [2, 0, -1, 4],
            [-3, 2, 1, -1],
        ],
        dtype=np.int8,
    )
    return GptqProjection(
        name="q_proj",
        rows=3,
        cols=4,
        group_size=4,
        qweight=pack_int4(weights, signed=True),
        scales=(1.0, 1.0, 1.0),
        zeros=(0, 0, 0),
        signed=True,
    )


def projection_to_report(proj: GptqProjection) -> dict[str, Any]:
    unpacked = unpack_int4(proj.qweight, proj.rows * proj.cols, signed=proj.signed).reshape(proj.rows, proj.cols)
    data = asdict(proj)
    data["qweight"] = list(proj.qweight)
    data["unpacked_int4"] = unpacked.astype(int).tolist()
    return data


def write_gptq_report(proj: GptqProjection, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gptq_weight_report.json"
    path.write_text(json.dumps(projection_to_report(proj), indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _candidate_metadata_files(model_dir: Path) -> list[Path]:
    names = [
        "quantize_config.json",
        "quant_config.json",
        "gptq_config.json",
        "quantization_config.json",
        "config.json",
    ]
    return [model_dir / name for name in names if (model_dir / name).exists()]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "none", "null"}:
            return None
        match = re.search(r"-?\d+", stripped)
        if match:
            return int(match.group(0))
    return None


def _extract_quant_config(raw: dict[str, Any]) -> dict[str, Any]:
    quant = raw.get("quantization_config", raw)
    if not isinstance(quant, dict):
        quant = raw
    bits = quant.get("bits", quant.get("w_bit", quant.get("weight_bits", quant.get("bits_per_weight"))))
    group_size = quant.get("group_size", quant.get("q_group_size", quant.get("groupsize")))
    return {
        "bits": _int_or_none(bits),
        "group_size": _int_or_none(group_size),
        "quant_method": quant.get("quant_method"),
        "desc_act": quant.get("desc_act", quant.get("act_order")),
        "sym": quant.get("sym"),
        "true_sequential": quant.get("true_sequential"),
        "checkpoint_format": quant.get("checkpoint_format", quant.get("format")),
        "meta": {
            key: value
            for key, value in quant.items()
            if key
            not in {
                "bits",
                "w_bit",
                "weight_bits",
                "bits_per_weight",
                "group_size",
                "q_group_size",
                "groupsize",
                "quant_method",
            }
        },
    }


def _projection_name_from_key(key: str) -> str | None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if f".{name}." in key or key.startswith(f"{name}."):
            return name
    return None


def _projection_entry() -> dict[str, Any]:
    return {
        "qweight": [],
        "qzeros": [],
        "scales": [],
        "g_idx": [],
        "other": [],
        "tensor_summaries": {},
    }


def _tensor_kind_from_key(key: str) -> str:
    suffix = key.rsplit(".", 1)[-1]
    if suffix in {"qweight", "q_weight", "qweight_packed"}:
        return "qweight"
    if suffix in {"qzeros", "q_zero", "qzeros_packed", "zeros", "zero_points"}:
        return "qzeros"
    if suffix in {"scales", "scale"}:
        return "scales"
    if suffix in {"g_idx", "gidx"}:
        return "g_idx"
    return "other"


def _record_projection_key(
    projection_keys: dict[str, dict[str, Any]],
    key: str,
    tensor_summary: dict[str, Any] | None = None,
) -> None:
    projection = _projection_name_from_key(key)
    if projection is None:
        return
    entry = projection_keys.setdefault(projection, _projection_entry())
    kind = _tensor_kind_from_key(key)
    if kind == "other":
        entry["other"].append(key)
    else:
        entry[kind].append(key)
    if tensor_summary is not None:
        entry["tensor_summaries"][key] = tensor_summary


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"{path.name} is too small to contain a safetensors header")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"{path.name} has an invalid safetensors header length")
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path.name} ended before the safetensors header completed")
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} safetensors header must contain a JSON object")
    return data


def _read_safetensors_header_and_data_start(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"{path.name} is too small to contain a safetensors header")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"{path.name} has an invalid safetensors header length")
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path.name} ended before the safetensors header completed")
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} safetensors header must contain a JSON object")
    return data, 8 + int(header_len)


def _safetensors_tensor_summary(path: Path, key: str, raw: Any) -> dict[str, Any]:
    return _safetensors_tensor_summary_from_file_name(path.name, key, raw)


def _safetensors_tensor_summary_from_file_name(file_name: str, key: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"file": file_name, "key": key, "metadata_status": "non_object_header_entry"}
    offsets = raw.get("data_offsets")
    byte_count = None
    if (
        isinstance(offsets, list)
        and len(offsets) == 2
        and isinstance(offsets[0], int)
        and isinstance(offsets[1], int)
        and offsets[1] >= offsets[0]
    ):
        byte_count = offsets[1] - offsets[0]
    return {
        "file": file_name,
        "key": key,
        "dtype": raw.get("dtype"),
        "shape": raw.get("shape") if isinstance(raw.get("shape"), list) else None,
        "data_offsets": offsets if isinstance(offsets, list) else None,
        "byte_count": byte_count,
        "metadata_status": "header_only_no_tensor_payload",
    }


def _inspect_weight_index(model_dir: Path) -> dict[str, Any]:
    index_paths = sorted(model_dir.glob("*.safetensors.index.json")) + sorted(model_dir.glob("*.bin.index.json"))
    projection_keys: dict[str, dict[str, Any]] = {}
    files_seen: set[str] = set()
    header_cache: dict[str, dict[str, Any] | None] = {}
    indexed_safetensors_files: list[str] = []
    indexed_safetensors_errors: list[dict[str, str]] = []
    direct_safetensors_files: list[str] = []
    direct_safetensors_errors: list[dict[str, str]] = []

    def summary_from_index_reference(filename: str, key: str) -> dict[str, Any] | None:
        if not filename.endswith(".safetensors"):
            return None
        tensor_path = model_dir / filename
        if filename not in header_cache:
            if not tensor_path.exists():
                header_cache[filename] = None
                indexed_safetensors_errors.append(
                    {
                        "file": filename,
                        "error": "referenced safetensors shard was not found",
                    }
                )
                return None
            try:
                header_cache[filename] = _read_safetensors_header(tensor_path)
                indexed_safetensors_files.append(filename)
            except Exception as exc:
                header_cache[filename] = None
                indexed_safetensors_errors.append(
                    {
                        "file": filename,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return None
        header = header_cache[filename]
        if header is None:
            return None
        raw_tensor_metadata = header.get(key)
        if raw_tensor_metadata is None:
            indexed_safetensors_errors.append(
                {
                    "file": filename,
                    "key": key,
                    "error": "referenced tensor key was not found in safetensors header",
                }
            )
            return None
        return _safetensors_tensor_summary(tensor_path, key, raw_tensor_metadata)

    for index_path in index_paths:
        data = _read_json(index_path)
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            continue
        for key, filename in weight_map.items():
            if isinstance(filename, str):
                files_seen.add(filename)
            projection = _projection_name_from_key(str(key))
            if projection is None:
                continue
            tensor_summary = summary_from_index_reference(filename, str(key)) if isinstance(filename, str) else None
            _record_projection_key(projection_keys, str(key), tensor_summary)
    if not index_paths:
        for tensor_path in sorted(model_dir.glob("*.safetensors")):
            direct_safetensors_files.append(tensor_path.name)
            try:
                header = _read_safetensors_header(tensor_path)
                for key, raw_tensor_metadata in header.items():
                    if key == "__metadata__":
                        continue
                    _record_projection_key(
                        projection_keys,
                        str(key),
                        _safetensors_tensor_summary(tensor_path, str(key), raw_tensor_metadata),
                    )
            except Exception as exc:
                direct_safetensors_errors.append(
                    {
                        "file": tensor_path.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    projections = []
    tensor_summary_count = 0
    for name, keys in sorted(projection_keys.items()):
        tensor_summary_count += len(keys["tensor_summaries"])
        projections.append(
            {
                "name": name,
                "has_qweight": bool(keys["qweight"]),
                "has_qzeros": bool(keys["qzeros"]),
                "has_scales": bool(keys["scales"]),
                "has_g_idx": bool(keys["g_idx"]),
                "keys": {
                    "qweight": keys["qweight"],
                    "qzeros": keys["qzeros"],
                    "scales": keys["scales"],
                    "g_idx": keys["g_idx"],
                    "other": keys["other"],
                },
                "tensor_summaries": keys["tensor_summaries"],
            }
        )
    quantized_projection_count = sum(1 for projection in projections if projection["has_qweight"])
    complete_gptq_projection_count = sum(
        1
        for projection in projections
        if projection["has_qweight"] and projection["has_qzeros"] and projection["has_scales"]
    )
    return {
        "index_files": [path.name for path in index_paths],
        "indexed_safetensors_files_scanned": sorted(set(indexed_safetensors_files)),
        "indexed_safetensors_header_errors": indexed_safetensors_errors,
        "direct_safetensors_files_scanned": direct_safetensors_files,
        "direct_safetensors_header_errors": direct_safetensors_errors,
        "tensor_key_source": "weight_index" if index_paths else ("safetensors_header" if direct_safetensors_files else "not_observed"),
        "shard_files_referenced": sorted(files_seen),
        "projection_metadata": projections,
        "projection_metadata_count": len(projections),
        "quantized_projection_metadata_count": quantized_projection_count,
        "complete_gptq_projection_metadata_count": complete_gptq_projection_count,
        "tensor_summary_count": tensor_summary_count,
        "tensor_summary_source": (
            "safetensors_header_from_weight_index"
            if tensor_summary_count and index_paths
            else ("safetensors_header" if tensor_summary_count else "not_available")
        ),
    }


def _checkpoint_quantization_artifact_report(
    *,
    status: str,
    index_report: dict[str, Any] | None = None,
    bits: int | None = None,
    group_size: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    index_report = index_report or {}
    raw_projection_count = int(index_report.get("projection_metadata_count", 0) or 0)
    quantized_projection_count = int(index_report.get("quantized_projection_metadata_count", 0) or 0)
    complete_projection_count = int(index_report.get("complete_gptq_projection_metadata_count", 0) or 0)
    has_tensor_headers = index_report.get("tensor_key_source") in {"weight_index", "safetensors_header"}
    has_plain_projection_weights = raw_projection_count > 0 and quantized_projection_count == 0

    if status == "parsed":
        if bits == 4 and isinstance(group_size, int) and group_size > 0 and complete_projection_count > 0:
            classification = "candidate_gptq_int4_checkpoint"
            dependency = "satisfied_by_gptq_quant_metadata"
            next_action = "run target layout and payload probes for all required LLaMA projections"
        elif bits != 4:
            classification = "unsupported_quantization_bits"
            dependency = "blocked_by_non_int4_quantization_metadata"
            next_action = "provide a GPTQ INT4 checkpoint with bits=4 and groupwise qzeros/scales"
        elif not isinstance(group_size, int) or group_size <= 0:
            classification = "missing_or_invalid_group_size"
            dependency = "blocked_by_missing_gptq_group_size"
            next_action = "provide GPTQ metadata with a positive group_size"
        else:
            classification = "gptq_config_without_complete_projection_tensors"
            dependency = "blocked_by_incomplete_gptq_projection_metadata"
            next_action = "provide qweight, qzeros, and scales tensor metadata for every required projection"
    elif status == "metadata_json_without_quant_fields" and has_plain_projection_weights:
        classification = "base_or_unquantized_checkpoint"
        dependency = "blocked_by_non_gptq_checkpoint_source"
        next_action = "set model.gptq_checkpoint to a GPTQ INT4 checkpoint rather than a base LLaMA checkpoint"
    elif status == "metadata_json_without_quant_fields" and quantized_projection_count > 0:
        classification = "gptq_tensor_keys_without_quant_config"
        dependency = "blocked_by_missing_gptq_quant_config"
        next_action = "add quantize_config.json, quant_config.json, gptq_config.json, or quantization_config with bits/group_size"
    elif status == "metadata_json_without_quant_fields":
        classification = "metadata_json_without_quantization_fields"
        dependency = "blocked_by_missing_gptq_quant_config"
        next_action = "provide checkpoint metadata that declares GPTQ bits and group_size"
    elif has_tensor_headers:
        classification = "tensor_headers_without_quantization_metadata"
        dependency = "blocked_by_missing_gptq_quant_config"
        next_action = "provide GPTQ quantization metadata and complete qweight/qzeros/scales tensors"
    elif status == "unavailable":
        classification = "checkpoint_source_unavailable"
        dependency = "blocked_by_checkpoint_source_preflight"
        next_action = "provide a local GPTQ checkpoint path or a populated Hugging Face cache entry"
    else:
        classification = "checkpoint_quantization_unknown"
        dependency = "blocked_by_missing_gptq_quant_config"
        next_action = "provide a GPTQ INT4 checkpoint with quantization metadata"

    return {
        "status": status,
        "classification": classification,
        "target_quantization": "gptq_int4",
        "checkpoint_quantization_dependency": dependency,
        "bits": bits,
        "group_size": group_size,
        "raw_projection_metadata_count": raw_projection_count,
        "quantized_projection_metadata_count": quantized_projection_count,
        "complete_gptq_projection_metadata_count": complete_projection_count,
        "has_tensor_headers": has_tensor_headers,
        "has_plain_projection_weights": has_plain_projection_weights,
        "reason": reason,
        "next_action": next_action,
        "does_not_claim": [
            "numeric_GPTQ_correctness",
            "checkpoint_tensor_payload_loading",
            "full_LLaMA_execution",
        ],
    }


def _checkpoint_root_from_local_path(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _artifact_inventory(model_dir: Path) -> dict[str, Any]:
    if not model_dir.exists() or not model_dir.is_dir():
        return {
            "model_dir": str(model_dir),
            "file_count": 0,
            "files": [],
            "matched_patterns": [],
            "missing_metadata_json": True,
            "has_weight_index": False,
            "has_safetensors": False,
        }
    files: list[str] = []
    matched_patterns: set[str] = set()
    for path in sorted(item for item in model_dir.iterdir() if item.is_file()):
        for pattern in GPTQ_CHECKPOINT_ALLOW_PATTERNS:
            if fnmatch.fnmatch(path.name, pattern):
                files.append(path.name)
                matched_patterns.add(pattern)
                break
    metadata_names = {
        "config.json",
        "quantize_config.json",
        "quant_config.json",
        "gptq_config.json",
        "quantization_config.json",
    }
    return {
        "model_dir": str(model_dir),
        "file_count": len(files),
        "files": files,
        "matched_patterns": sorted(matched_patterns),
        "missing_metadata_json": not any(name in files for name in metadata_names),
        "has_weight_index": any(name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json") for name in files),
        "has_safetensors": any(name.endswith(".safetensors") for name in files),
    }


def _artifact_inventory_from_names(source: str, names: list[str]) -> dict[str, Any]:
    files: list[str] = []
    matched_patterns: set[str] = set()
    for name in sorted(names):
        file_name = Path(name).name
        for pattern in GPTQ_CHECKPOINT_ALLOW_PATTERNS:
            if fnmatch.fnmatch(file_name, pattern):
                files.append(name)
                matched_patterns.add(pattern)
                break
    metadata_names = {
        "config.json",
        "quantize_config.json",
        "quant_config.json",
        "gptq_config.json",
        "quantization_config.json",
    }
    return {
        "source": source,
        "file_count": len(files),
        "files": files,
        "matched_patterns": sorted(matched_patterns),
        "missing_metadata_json": not any(Path(name).name in metadata_names for name in files),
        "has_weight_index": any(name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json") for name in files),
        "has_safetensors": any(name.endswith(".safetensors") for name in files),
    }


def _inventory_has_checkpoint_payload(inventory: dict[str, Any]) -> bool:
    return (
        int(inventory.get("file_count", 0) or 0) > 0
        and inventory.get("missing_metadata_json") is False
        and (inventory.get("has_weight_index") is True or inventory.get("has_safetensors") is True)
    )


def _remote_preflight_allowed() -> bool:
    return os.environ.get("NL2HDL_ALLOW_HF_REMOTE_PREFLIGHT") == "1"


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _hf_auth_headers() -> dict[str, str]:
    token = _hf_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _hf_remote_list_repo_files(repo_id: str, revision: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo_id=repo_id, revision=revision))


def _hf_remote_read_json(repo_id: str, filename: str, revision: str | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_files_only=False,
    )
    return _read_json(Path(path))


def _hf_remote_get_bytes(
    repo_id: str,
    filename: str,
    start: int,
    end_inclusive: int,
    revision: str | None = None,
) -> bytes:
    if start < 0 or end_inclusive < start:
        raise ValueError("invalid remote byte range")
    try:
        import requests
        from huggingface_hub import hf_hub_url
    except Exception as exc:
        raise RuntimeError(f"remote range read requires requests and huggingface_hub: {exc}") from exc

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    headers = {"Range": f"bytes={start}-{end_inclusive}", **_hf_auth_headers()}
    expected = end_inclusive - start + 1
    with requests.get(url, headers=headers, timeout=30, stream=True) as response:
        if response.status_code not in {200, 206}:
            raise RuntimeError(f"HTTP {response.status_code} while reading {filename}")
        if start > 0 and response.status_code != 206:
            raise RuntimeError(f"remote host did not honor Range for {filename}")
        chunks: list[bytes] = []
        remaining = expected
        for chunk in response.iter_content(chunk_size=min(65536, expected)):
            if not chunk:
                continue
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                remaining = 0
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining == 0:
                break
    payload = b"".join(chunks)
    if len(payload) != expected:
        raise RuntimeError(
            f"short remote range read for {filename}: requested {expected} bytes, observed {len(payload)}"
        )
    return payload


def _read_remote_safetensors_header(repo_id: str, filename: str, revision: str | None = None) -> dict[str, Any]:
    header_len_raw = _hf_remote_get_bytes(repo_id, filename, 0, 7, revision=revision)
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or header_len > 256 * 1024 * 1024:
        raise ValueError(f"{filename} has an invalid safetensors header length")
    header = _hf_remote_get_bytes(repo_id, filename, 8, 8 + int(header_len) - 1, revision=revision)
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{filename} safetensors header must contain a JSON object")
    return data


def _read_remote_safetensors_header_and_data_start(
    repo_id: str,
    filename: str,
    revision: str | None = None,
) -> tuple[dict[str, Any], int]:
    header_len_raw = _hf_remote_get_bytes(repo_id, filename, 0, 7, revision=revision)
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or header_len > 256 * 1024 * 1024:
        raise ValueError(f"{filename} has an invalid safetensors header length")
    header = _hf_remote_get_bytes(repo_id, filename, 8, 8 + int(header_len) - 1, revision=revision)
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{filename} safetensors header must contain a JSON object")
    return data, 8 + int(header_len)


def _candidate_remote_metadata_files(files: list[str]) -> list[str]:
    names = [
        "quantize_config.json",
        "quant_config.json",
        "gptq_config.json",
        "quantization_config.json",
        "config.json",
    ]
    available = {Path(name).name: name for name in files}
    return [available[name] for name in names if name in available]


def _inspect_remote_weight_index(
    repo_id: str,
    files: list[str],
    revision: str | None = None,
) -> dict[str, Any]:
    index_paths = sorted(name for name in files if name.endswith(".safetensors.index.json")) + sorted(
        name for name in files if name.endswith(".bin.index.json")
    )
    projection_keys: dict[str, dict[str, Any]] = {}
    files_seen: set[str] = set()
    header_cache: dict[str, dict[str, Any] | None] = {}
    indexed_safetensors_files: list[str] = []
    indexed_safetensors_errors: list[dict[str, str]] = []
    direct_safetensors_files: list[str] = []
    direct_safetensors_errors: list[dict[str, str]] = []
    index_errors: list[dict[str, str]] = []
    file_set = set(files)

    def summary_from_index_reference(filename: str, key: str) -> dict[str, Any] | None:
        if not filename.endswith(".safetensors"):
            return None
        if filename not in file_set:
            header_cache[filename] = None
            indexed_safetensors_errors.append(
                {
                    "file": filename,
                    "error": "referenced safetensors shard was not found in remote repo file list",
                }
            )
            return None
        if filename not in header_cache:
            try:
                header_cache[filename] = _read_remote_safetensors_header(repo_id, filename, revision=revision)
                indexed_safetensors_files.append(filename)
            except Exception as exc:
                header_cache[filename] = None
                indexed_safetensors_errors.append(
                    {
                        "file": filename,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return None
        header = header_cache[filename]
        if header is None:
            return None
        raw_tensor_metadata = header.get(key)
        if raw_tensor_metadata is None:
            indexed_safetensors_errors.append(
                {
                    "file": filename,
                    "key": key,
                    "error": "referenced tensor key was not found in safetensors header",
                }
            )
            return None
        return _safetensors_tensor_summary_from_file_name(filename, key, raw_tensor_metadata)

    for index_path in index_paths:
        try:
            data = _hf_remote_read_json(repo_id, index_path, revision=revision)
        except Exception as exc:
            index_errors.append({"file": index_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            continue
        for key, filename in weight_map.items():
            if isinstance(filename, str):
                files_seen.add(filename)
            projection = _projection_name_from_key(str(key))
            if projection is None:
                continue
            tensor_summary = summary_from_index_reference(filename, str(key)) if isinstance(filename, str) else None
            _record_projection_key(projection_keys, str(key), tensor_summary)
    if not index_paths:
        limit = _int_or_none(os.environ.get("NL2HDL_HF_REMOTE_HEADER_FILE_LIMIT")) or 32
        for filename in sorted(name for name in files if name.endswith(".safetensors"))[:limit]:
            direct_safetensors_files.append(filename)
            try:
                header = _read_remote_safetensors_header(repo_id, filename, revision=revision)
                for key, raw_tensor_metadata in header.items():
                    if key == "__metadata__":
                        continue
                    _record_projection_key(
                        projection_keys,
                        str(key),
                        _safetensors_tensor_summary_from_file_name(filename, str(key), raw_tensor_metadata),
                    )
            except Exception as exc:
                direct_safetensors_errors.append(
                    {
                        "file": filename,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    projections = []
    tensor_summary_count = 0
    for name, keys in sorted(projection_keys.items()):
        tensor_summary_count += len(keys["tensor_summaries"])
        projections.append(
            {
                "name": name,
                "has_qweight": bool(keys["qweight"]),
                "has_qzeros": bool(keys["qzeros"]),
                "has_scales": bool(keys["scales"]),
                "has_g_idx": bool(keys["g_idx"]),
                "keys": {
                    "qweight": keys["qweight"],
                    "qzeros": keys["qzeros"],
                    "scales": keys["scales"],
                    "g_idx": keys["g_idx"],
                    "other": keys["other"],
                },
                "tensor_summaries": keys["tensor_summaries"],
            }
        )
    quantized_projection_count = sum(1 for projection in projections if projection["has_qweight"])
    complete_gptq_projection_count = sum(
        1
        for projection in projections
        if projection["has_qweight"] and projection["has_qzeros"] and projection["has_scales"]
    )
    return {
        "index_files": index_paths,
        "index_file_errors": index_errors,
        "indexed_safetensors_files_scanned": sorted(set(indexed_safetensors_files)),
        "indexed_safetensors_header_errors": indexed_safetensors_errors,
        "direct_safetensors_files_scanned": direct_safetensors_files,
        "direct_safetensors_header_errors": direct_safetensors_errors,
        "tensor_key_source": "weight_index" if index_paths else ("safetensors_header" if direct_safetensors_files else "not_observed"),
        "shard_files_referenced": sorted(files_seen),
        "projection_metadata": projections,
        "projection_metadata_count": len(projections),
        "quantized_projection_metadata_count": quantized_projection_count,
        "complete_gptq_projection_metadata_count": complete_gptq_projection_count,
        "tensor_summary_count": tensor_summary_count,
        "tensor_summary_source": (
            "remote_safetensors_header_from_weight_index"
            if tensor_summary_count and index_paths
            else ("remote_safetensors_header" if tensor_summary_count else "not_available")
        ),
        "remote_repo_id": repo_id,
        "remote_revision": revision,
    }


def build_gptq_checkpoint_source_preflight_report(model_name: str) -> dict[str, Any]:
    allow_network_snapshot = os.environ.get("NL2HDL_ALLOW_HF_SNAPSHOT_DOWNLOAD") == "1"
    allow_remote_lightweight = _remote_preflight_allowed()
    report: dict[str, Any] = {
        "artifact": "gptq_checkpoint_source_preflight",
        "model_name": model_name,
        "status": "unresolved",
        "checkpoint_source_dependency": "blocked_by_checkpoint_source_preflight",
        "expected_file_patterns": GPTQ_CHECKPOINT_ALLOW_PATTERNS,
        "network_snapshot_download_allowed": allow_network_snapshot,
        "network_snapshot_download_env": "NL2HDL_ALLOW_HF_SNAPSHOT_DOWNLOAD",
        "remote_lightweight_preflight_allowed": allow_remote_lightweight,
        "remote_lightweight_preflight_env": "NL2HDL_ALLOW_HF_REMOTE_PREFLIGHT",
        "local_path_probe": {},
        "huggingface_hub_probe": {},
        "local_cache_probe": {},
        "remote_lightweight_probe": {
            "attempted": False,
            "reason": "disabled_by_default",
        },
        "network_download_probe": {
            "attempted": False,
            "reason": "disabled_by_default",
        },
        "next_action": "provide a local model.gptq_checkpoint path or pre-populate the Hugging Face cache",
        "does_not_claim": [
            "permission_to_download_large_checkpoints_by_default",
            "checkpoint_weight_loading",
            "full_tensor_materialization",
            "numeric_GPTQ_correctness",
            "full_LLaMA_execution",
        ],
    }

    requested_path = Path(model_name).expanduser()
    local_path_exists = requested_path.exists()
    report["local_path_probe"] = {
        "path": str(requested_path),
        "exists": local_path_exists,
        "is_dir": requested_path.is_dir() if local_path_exists else False,
        "is_file": requested_path.is_file() if local_path_exists else False,
    }
    if local_path_exists:
        model_dir = _checkpoint_root_from_local_path(requested_path)
        report["status"] = "resolved_local_path"
        report["checkpoint_source_dependency"] = "satisfied_by_local_path"
        report["resolved_model_dir"] = str(model_dir)
        report["artifact_inventory"] = _artifact_inventory(model_dir)
        report["next_action"] = "run GPTQ metadata and payload probes against the resolved local checkpoint directory"
        return report

    try:
        from huggingface_hub import snapshot_download

        report["huggingface_hub_probe"] = {"available": True}
    except Exception as exc:
        report["huggingface_hub_probe"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["blocking_reason"] = "huggingface_hub is not available and no local checkpoint path exists"
        report["next_action"] = "install huggingface_hub or set model.gptq_checkpoint to an existing local directory"
        return report

    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            allow_patterns=GPTQ_CHECKPOINT_ALLOW_PATTERNS,
        )
        model_dir = Path(snapshot_path)
        report["status"] = "resolved_huggingface_local_cache"
        report["checkpoint_source_dependency"] = "satisfied_by_huggingface_local_cache"
        report["resolved_model_dir"] = str(model_dir)
        inventory = _artifact_inventory(model_dir)
        report["local_cache_probe"] = {
            "attempted": True,
            "status": "resolved" if _inventory_has_checkpoint_payload(inventory) else "incomplete",
            "path": snapshot_path,
            "artifact_inventory": inventory,
        }
        if _inventory_has_checkpoint_payload(inventory):
            report["artifact_inventory"] = inventory
            report["next_action"] = "run GPTQ metadata and payload probes against the resolved Hugging Face cache directory"
            return report
        report["status"] = "unresolved"
        report["checkpoint_source_dependency"] = "blocked_by_checkpoint_source_preflight"
        report["blocking_reason"] = (
            "Hugging Face local cache exists but lacks safetensors/index files required for GPTQ target preflight"
        )
    except Exception as exc:
        report["local_cache_probe"] = {
            "attempted": True,
            "status": "unresolved",
            "error": f"{type(exc).__name__}: {exc}",
            }

    if allow_remote_lightweight:
        try:
            files = _hf_remote_list_repo_files(model_name)
            inventory = _artifact_inventory_from_names(model_name, files)
            report["remote_lightweight_probe"] = {
                "attempted": True,
                "status": "resolved",
                "repo_id": model_name,
                "file_count": len(files),
                "files": files,
                "artifact_inventory": inventory,
            }
            if inventory["file_count"] > 0 and not inventory["missing_metadata_json"] and (
                inventory["has_weight_index"] or inventory["has_safetensors"]
            ):
                report["status"] = "resolved_huggingface_remote_lightweight"
                report["checkpoint_source_dependency"] = "satisfied_by_huggingface_remote_lightweight_preflight"
                report["resolved_remote_repo_id"] = model_name
                report["artifact_inventory"] = inventory
                report["next_action"] = (
                    "run remote lightweight GPTQ metadata/header and bounded payload probes without full checkpoint download"
                )
                return report
            report["remote_lightweight_probe"]["status"] = "incomplete"
            report["remote_lightweight_probe"]["reason"] = (
                "remote repo did not expose both quantization metadata and safetensors/index files"
            )
        except Exception as exc:
            report["remote_lightweight_probe"] = {
                "attempted": True,
                "status": "unresolved",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if allow_network_snapshot:
        try:
            snapshot_path = snapshot_download(
                repo_id=model_name,
                local_files_only=False,
                allow_patterns=GPTQ_CHECKPOINT_ALLOW_PATTERNS,
            )
            model_dir = Path(snapshot_path)
            report["status"] = "resolved_huggingface_snapshot_download"
            report["checkpoint_source_dependency"] = "satisfied_by_explicit_huggingface_snapshot_download"
            report["resolved_model_dir"] = str(model_dir)
            report["network_download_probe"] = {
                "attempted": True,
                "status": "resolved",
                "path": snapshot_path,
            }
            report["artifact_inventory"] = _artifact_inventory(model_dir)
            report["next_action"] = "run GPTQ metadata and payload probes against the downloaded snapshot"
            return report
        except Exception as exc:
            report["network_download_probe"] = {
                "attempted": True,
                "status": "unresolved",
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["blocking_reason"] = "explicit Hugging Face snapshot download did not resolve the checkpoint"
            report["next_action"] = (
                "check repository access, accept gated model terms if required, authenticate with Hugging Face, "
                "or provide a local model.gptq_checkpoint path"
            )
            return report

    report["blocking_reason"] = "no local path or Hugging Face local cache entry was resolved; network snapshot download is disabled"
    report["next_action"] = (
        "pre-populate the Hugging Face cache, set model.gptq_checkpoint to a local path, or explicitly set "
        "NL2HDL_ALLOW_HF_SNAPSHOT_DOWNLOAD=1 for a bounded metadata/checkpoint snapshot attempt"
    )
    return report


def _resolve_local_model_dir(model_name: str) -> tuple[Path | None, dict[str, Any]]:
    path = Path(model_name).expanduser()
    if path.exists():
        return _checkpoint_root_from_local_path(path), {"source": "local_path", "path": str(path)}
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            allow_patterns=GPTQ_CHECKPOINT_ALLOW_PATTERNS,
        )
        model_dir = Path(snapshot_path)
        inventory = _artifact_inventory(model_dir)
        if _inventory_has_checkpoint_payload(inventory):
            return model_dir, {
                "source": "huggingface_local_cache",
                "path": snapshot_path,
                "artifact_inventory": inventory,
            }
        return None, {
            "source": "unresolved",
            "reason": "Hugging Face local cache is incomplete for GPTQ target preflight",
            "path": snapshot_path,
            "artifact_inventory": inventory,
        }
    except Exception as exc:
        return None, {"source": "unresolved", "reason": f"{type(exc).__name__}: {exc}"}


def _resolve_remote_checkpoint_from_preflight(source_preflight: dict[str, Any]) -> dict[str, Any] | None:
    if source_preflight.get("status") != "resolved_huggingface_remote_lightweight":
        return None
    probe = source_preflight.get("remote_lightweight_probe")
    if not isinstance(probe, dict):
        return None
    files = probe.get("files")
    repo_id = source_preflight.get("resolved_remote_repo_id") or probe.get("repo_id")
    if not isinstance(repo_id, str) or not isinstance(files, list):
        return None
    return {
        "source": "huggingface_remote_lightweight",
        "repo_id": repo_id,
        "revision": probe.get("revision"),
        "files": [str(item) for item in files],
    }


def _inspect_remote_gptq_checkpoint_metadata(
    model_name: str,
    source_preflight: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    repo_id = str(resolution["repo_id"])
    revision = resolution.get("revision") if isinstance(resolution.get("revision"), str) else None
    files = [str(item) for item in resolution.get("files", [])]
    report: dict[str, Any] = {
        "artifact": "gptq_checkpoint_metadata",
        "model_name": model_name,
        "status": "unavailable",
        "checkpoint_source_preflight": source_preflight,
        "metadata_resolution": resolution,
        "network_download_allowed": source_preflight["network_snapshot_download_allowed"],
        "remote_lightweight_preflight_allowed": source_preflight.get("remote_lightweight_preflight_allowed"),
        "remote_repo_id": repo_id,
        "remote_revision": revision,
        "does_not_claim": [
            "checkpoint_weight_loading",
            "full_tensor_materialization",
            "numeric_GPTQ_correctness",
            "full_LLaMA_execution",
        ],
    }
    metadata_files = _candidate_remote_metadata_files(files)
    if not metadata_files:
        report["reason"] = "no GPTQ quantization metadata JSON file found in remote repo file list"
        index_report = _inspect_remote_weight_index(repo_id, files, revision=revision)
        report.update(index_report)
        report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
            status=report["status"],
            index_report=index_report,
            reason=report["reason"],
        )
        return report

    parsed_configs = []
    selected_quant: dict[str, Any] | None = None
    metadata_errors: list[dict[str, str]] = []
    for metadata_name in metadata_files:
        try:
            raw = _hf_remote_read_json(repo_id, metadata_name, revision=revision)
        except Exception as exc:
            metadata_errors.append({"file": metadata_name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        quant = _extract_quant_config(raw)
        parsed_configs.append({"file": Path(metadata_name).name, "remote_path": metadata_name, "quant_config": quant})
        if selected_quant is None and (quant["bits"] is not None or quant["group_size"] is not None):
            selected_quant = quant

    index_report = _inspect_remote_weight_index(repo_id, files, revision=revision)
    report.update(index_report)
    report["metadata_files"] = [Path(name).name for name in metadata_files]
    report["remote_metadata_files"] = metadata_files
    report["metadata_file_errors"] = metadata_errors
    report["parsed_configs"] = parsed_configs
    if selected_quant is None:
        report["status"] = "metadata_json_without_quant_fields"
        report["reason"] = "remote metadata JSON files exist but did not expose bits or group_size"
        if metadata_errors and not parsed_configs:
            report["status"] = "unavailable"
            report["reason"] = "remote metadata JSON files could not be read"
        report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
            status=report["status"],
            index_report=index_report,
            reason=report["reason"],
        )
        return report

    report["status"] = "parsed"
    report["bits"] = selected_quant["bits"]
    report["group_size"] = selected_quant["group_size"]
    report["quant_method"] = selected_quant["quant_method"]
    report["desc_act"] = selected_quant["desc_act"]
    report["sym"] = selected_quant["sym"]
    report["true_sequential"] = selected_quant["true_sequential"]
    report["checkpoint_format"] = selected_quant["checkpoint_format"]
    report["packing_order"] = "checkpoint_specific_qweight_layout_requires_loader_contract"
    report["scales_and_zero_points"] = {
        "source": "remote_tensor_key_presence" if index_report["projection_metadata"] else "not_observed_in_tensor_keys",
        "requires_tensor_loader_for_values": True,
    }
    report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
        status=report["status"],
        index_report=index_report,
        bits=report["bits"],
        group_size=report["group_size"],
    )
    return report


def inspect_gptq_checkpoint_metadata(model_name: str) -> dict[str, Any]:
    source_preflight = build_gptq_checkpoint_source_preflight_report(model_name)
    model_dir, resolution = _resolve_local_model_dir(model_name)
    if model_dir is None:
        remote_resolution = _resolve_remote_checkpoint_from_preflight(source_preflight)
        if remote_resolution is not None:
            return _inspect_remote_gptq_checkpoint_metadata(model_name, source_preflight, remote_resolution)
    report: dict[str, Any] = {
        "artifact": "gptq_checkpoint_metadata",
        "model_name": model_name,
        "status": "unavailable",
        "checkpoint_source_preflight": source_preflight,
        "metadata_resolution": resolution,
        "network_download_allowed": source_preflight["network_snapshot_download_allowed"],
        "does_not_claim": [
            "checkpoint_weight_loading",
            "full_tensor_materialization",
            "numeric_GPTQ_correctness",
            "full_LLaMA_execution",
        ],
    }
    if model_dir is None:
        report["reason"] = "no local checkpoint directory or Hugging Face cache metadata was found"
        report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
            status="unavailable",
            reason=report["reason"],
        )
        return report

    metadata_files = _candidate_metadata_files(model_dir)
    if not metadata_files:
        report["reason"] = "no GPTQ quantization metadata JSON file found"
        report["model_dir"] = str(model_dir)
        index_report = _inspect_weight_index(model_dir)
        report.update(index_report)
        report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
            status=report["status"],
            index_report=index_report,
            reason=report["reason"],
        )
        return report

    parsed_configs = []
    selected_quant: dict[str, Any] | None = None
    for metadata_path in metadata_files:
        raw = _read_json(metadata_path)
        quant = _extract_quant_config(raw)
        parsed_configs.append({"file": metadata_path.name, "quant_config": quant})
        if selected_quant is None and (quant["bits"] is not None or quant["group_size"] is not None):
            selected_quant = quant

    index_report = _inspect_weight_index(model_dir)
    report.update(index_report)
    report["model_dir"] = str(model_dir)
    report["metadata_files"] = [path.name for path in metadata_files]
    report["parsed_configs"] = parsed_configs
    if selected_quant is None:
        report["status"] = "metadata_json_without_quant_fields"
        report["reason"] = "metadata JSON files exist but did not expose bits or group_size"
        report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
            status=report["status"],
            index_report=index_report,
            reason=report["reason"],
        )
        return report

    report["status"] = "parsed"
    report["bits"] = selected_quant["bits"]
    report["group_size"] = selected_quant["group_size"]
    report["quant_method"] = selected_quant["quant_method"]
    report["desc_act"] = selected_quant["desc_act"]
    report["sym"] = selected_quant["sym"]
    report["true_sequential"] = selected_quant["true_sequential"]
    report["checkpoint_format"] = selected_quant["checkpoint_format"]
    report["packing_order"] = "checkpoint_specific_qweight_layout_requires_loader_contract"
    report["scales_and_zero_points"] = {
        "source": "tensor_key_presence" if index_report["projection_metadata"] else "not_observed_in_tensor_keys",
        "requires_tensor_loader_for_values": True,
    }
    report["checkpoint_quantization_artifact"] = _checkpoint_quantization_artifact_report(
        status=report["status"],
        index_report=index_report,
        bits=report["bits"],
        group_size=report["group_size"],
    )
    return report


def _projection_from_metadata(metadata: dict[str, Any], projection_name: str) -> dict[str, Any] | None:
    projections = metadata.get("projection_metadata", [])
    if not isinstance(projections, list):
        return None
    for projection in projections:
        if isinstance(projection, dict) and projection.get("name") == projection_name:
            return projection
    return None


def _first_summary_for_kind(projection: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
    if projection is None:
        return None
    summaries = projection.get("tensor_summaries")
    keys = projection.get("keys")
    if not isinstance(summaries, dict):
        return None
    if isinstance(keys, dict):
        for key in keys.get(kind, []):
            summary = summaries.get(key)
            if isinstance(summary, dict):
                result = dict(summary)
                result.setdefault("key", key)
                return result
    suffix = f".{kind}"
    for key, summary in summaries.items():
        if isinstance(key, str) and key.endswith(suffix) and isinstance(summary, dict):
            result = dict(summary)
            result.setdefault("key", key)
            return result
    return None


def _read_safetensors_tensor_prefix(model_dir: Path, summary: dict[str, Any], sample_bytes: int) -> dict[str, Any]:
    file_name = summary.get("file")
    key = summary.get("key")
    offsets = summary.get("data_offsets")
    byte_count = summary.get("byte_count")
    if not isinstance(file_name, str) or not file_name.endswith(".safetensors"):
        return {
            "status": "unavailable",
            "reason": "tensor summary does not identify a safetensors file",
            "key": key,
        }
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(offsets[0], int)
        or not isinstance(offsets[1], int)
        or offsets[1] < offsets[0]
    ):
        return {
            "status": "unavailable",
            "reason": "tensor summary does not contain valid safetensors data_offsets",
            "file": file_name,
            "key": key,
        }
    if not isinstance(byte_count, int):
        byte_count = offsets[1] - offsets[0]
    tensor_path = model_dir / file_name
    if not tensor_path.exists():
        return {
            "status": "unavailable",
            "reason": "safetensors file was not found",
            "file": file_name,
            "key": key,
        }
    try:
        header, data_start = _read_safetensors_header_and_data_start(tensor_path)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "file": file_name,
            "key": key,
        }
    if isinstance(key, str) and key not in header:
        return {
            "status": "unavailable",
            "reason": "tensor key was not found in safetensors header",
            "file": file_name,
            "key": key,
        }
    absolute_start = data_start + offsets[0]
    absolute_end = data_start + offsets[1]
    file_size = tensor_path.stat().st_size
    if absolute_end > file_size:
        return {
            "status": "unavailable",
            "reason": "tensor data_offsets exceed safetensors file size",
            "file": file_name,
            "key": key,
            "file_size": file_size,
            "absolute_end": absolute_end,
        }
    read_count = min(max(int(sample_bytes), 0), byte_count)
    with tensor_path.open("rb") as handle:
        handle.seek(absolute_start)
        payload = handle.read(read_count)
    if len(payload) != read_count:
        return {
            "status": "unavailable",
            "reason": "short read while loading tensor prefix",
            "file": file_name,
            "key": key,
            "requested_bytes": read_count,
            "observed_bytes": len(payload),
        }
    words32_le = [
        int.from_bytes(payload[idx : idx + 4].ljust(4, b"\0"), "little")
        for idx in range(0, len(payload), 4)
    ]
    return {
        "status": "sampled",
        "file": file_name,
        "key": key,
        "dtype": summary.get("dtype"),
        "shape": summary.get("shape"),
        "data_offsets": offsets,
        "byte_count": byte_count,
        "sample_start_offset": offsets[0],
        "sample_byte_count": len(payload),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "sample_bytes_hex": payload.hex(),
        "words32_le_hex": [f"0x{word:08x}" for word in words32_le],
        "source": "safetensors_payload_prefix",
    }


def _read_remote_safetensors_tensor_prefix(
    repo_id: str,
    summary: dict[str, Any],
    sample_bytes: int,
    revision: str | None = None,
) -> dict[str, Any]:
    file_name = summary.get("file")
    key = summary.get("key")
    offsets = summary.get("data_offsets")
    byte_count = summary.get("byte_count")
    if not isinstance(file_name, str) or not file_name.endswith(".safetensors"):
        return {
            "status": "unavailable",
            "reason": "tensor summary does not identify a safetensors file",
            "key": key,
        }
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(offsets[0], int)
        or not isinstance(offsets[1], int)
        or offsets[1] < offsets[0]
    ):
        return {
            "status": "unavailable",
            "reason": "tensor summary does not contain valid safetensors data_offsets",
            "file": file_name,
            "key": key,
        }
    if not isinstance(byte_count, int):
        byte_count = offsets[1] - offsets[0]
    try:
        header, data_start = _read_remote_safetensors_header_and_data_start(repo_id, file_name, revision=revision)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "file": file_name,
            "key": key,
            "remote_repo_id": repo_id,
        }
    if isinstance(key, str) and key not in header:
        return {
            "status": "unavailable",
            "reason": "tensor key was not found in safetensors header",
            "file": file_name,
            "key": key,
            "remote_repo_id": repo_id,
        }
    read_count = min(max(int(sample_bytes), 0), byte_count)
    absolute_start = data_start + offsets[0]
    absolute_end = absolute_start + read_count - 1
    try:
        payload = (
            b""
            if read_count == 0
            else _hf_remote_get_bytes(
                repo_id,
                file_name,
                absolute_start,
                absolute_end,
                revision=revision,
            )
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "file": file_name,
            "key": key,
            "remote_repo_id": repo_id,
            "absolute_start": absolute_start,
            "requested_bytes": read_count,
        }
    words32_le = [
        int.from_bytes(payload[idx : idx + 4].ljust(4, b"\0"), "little")
        for idx in range(0, len(payload), 4)
    ]
    return {
        "status": "sampled",
        "file": file_name,
        "key": key,
        "dtype": summary.get("dtype"),
        "shape": summary.get("shape"),
        "data_offsets": offsets,
        "byte_count": byte_count,
        "sample_start_offset": offsets[0],
        "sample_absolute_start_offset": absolute_start,
        "sample_byte_count": len(payload),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "sample_bytes_hex": payload.hex(),
        "words32_le_hex": [f"0x{word:08x}" for word in words32_le],
        "source": "remote_safetensors_payload_prefix",
        "remote_repo_id": repo_id,
        "remote_revision": revision,
    }


def _qweight_memory_beat_preview(
    words32_le_hex: list[str],
    sample_byte_count: int | None,
    memory_data_width_bits: int = 128,
) -> dict[str, Any]:
    words_per_beat = memory_data_width_bits // 32 if memory_data_width_bits > 0 else 0
    bytes_per_beat = memory_data_width_bits // 8 if memory_data_width_bits > 0 else 0
    full_beats = len(words32_le_hex) // words_per_beat if words_per_beat else 0
    beat_chunks = [
        words32_le_hex[idx : idx + words_per_beat]
        for idx in range(0, full_beats * words_per_beat, words_per_beat)
    ]
    beat_hex = []
    for chunk in beat_chunks:
        values = [int(word, 16) for word in chunk]
        beat_hex.append("0x" + "".join(f"{value:08x}" for value in reversed(values)))
    return {
        "source": "qweight_safetensors_payload_prefix",
        "payload_order": "safetensors_payload_prefix_32bit_little_endian_words",
        "memory_data_width_bits": memory_data_width_bits,
        "bytes_per_memory_beat": bytes_per_beat,
        "words32_per_memory_beat": words_per_beat,
        "sample_byte_count": sample_byte_count,
        "sampled_full_memory_beat_count": full_beats,
        "covers_first_memory_beat": full_beats >= 1,
        "first_memory_beats_128b_le_hex": beat_hex if memory_data_width_bits == 128 else [],
        "memory_beat_word_chunks32_le_hex": beat_chunks,
        "does_not_claim": [
            "full_qweight_payload_streaming",
            "AXI_or_DDR_burst_execution",
            "checkpoint_specific_qweight_order_correctness",
        ],
    }


def build_gptq_payload_probe_report(
    model_name: str,
    projection_name: str = "q_proj",
    sample_bytes: int = 64,
) -> dict[str, Any]:
    metadata = inspect_gptq_checkpoint_metadata(model_name)
    model_dir_value = metadata.get("model_dir")
    metadata_resolution = metadata.get("metadata_resolution")
    remote_repo_id = metadata_resolution.get("repo_id") if isinstance(metadata_resolution, dict) else None
    remote_revision = metadata_resolution.get("revision") if isinstance(metadata_resolution, dict) else None
    projection = _projection_from_metadata(metadata, projection_name)
    report: dict[str, Any] = {
        "artifact": "gptq_payload_probe",
        "model_name": model_name,
        "projection": projection_name,
        "sample_bytes_requested": sample_bytes,
        "checkpoint_metadata_status": metadata.get("status"),
        "metadata_resolution": metadata.get("metadata_resolution"),
        "status": "unavailable",
        "tensors": {},
        "qweight_payload_order": "safetensors_payload_prefix_32bit_little_endian_words",
        "qweight_payload_words32_le_hex": [],
        "qweight_payload_word_count": 0,
        "qweight_stream_probe": _qweight_memory_beat_preview([], None),
        "sampled_tensor_count": 0,
        "required_tensor_count": 3,
        "target_checkpoint_payload_dependency": "blocked_by_gptq_payload_probe",
        "does_not_claim": [
            "full_checkpoint_tensor_materialization",
            "numeric_GPTQ_correctness",
            "checkpoint_specific_qweight_order_correctness",
            "full_qweight_payload_streaming",
            "full_LLaMA_execution",
        ],
    }
    local_payload_source = isinstance(model_dir_value, str)
    remote_payload_source = (
        isinstance(metadata_resolution, dict)
        and metadata_resolution.get("source") == "huggingface_remote_lightweight"
        and isinstance(remote_repo_id, str)
    )
    if not local_payload_source and not remote_payload_source:
        report["reason"] = "checkpoint metadata did not resolve a local model_dir or remote lightweight repo"
        return report
    if projection is None:
        report["reason"] = "selected projection metadata was not found"
        return report
    model_dir = Path(model_dir_value) if isinstance(model_dir_value, str) else None
    sampled = 0
    for kind in ("qweight", "qzeros", "scales"):
        summary = _first_summary_for_kind(projection, kind)
        if summary is None:
            report["tensors"][kind] = {
                "status": "unavailable",
                "reason": f"{kind} tensor summary was not found",
            }
            continue
        if model_dir is not None:
            tensor_report = _read_safetensors_tensor_prefix(model_dir, summary, sample_bytes)
        else:
            tensor_report = _read_remote_safetensors_tensor_prefix(
                remote_repo_id,
                summary,
                sample_bytes,
                revision=remote_revision if isinstance(remote_revision, str) else None,
            )
        report["tensors"][kind] = tensor_report
        if tensor_report["status"] == "sampled":
            sampled += 1
    qweight_report = report["tensors"].get("qweight", {})
    report["qweight_payload_words32_le_hex"] = qweight_report.get("words32_le_hex", [])
    report["qweight_payload_word_count"] = len(report["qweight_payload_words32_le_hex"])
    report["qweight_stream_probe"] = _qweight_memory_beat_preview(
        report["qweight_payload_words32_le_hex"],
        qweight_report.get("sample_byte_count"),
    )
    report["sampled_tensor_count"] = sampled
    report["required_tensor_count"] = 3
    report["status"] = "sampled" if sampled == 3 else "partial" if sampled else "unavailable"
    report["target_checkpoint_payload_dependency"] = (
        "satisfied_by_payload_probe" if report["status"] == "sampled" else "blocked_by_gptq_payload_probe"
    )
    if report["status"] != "sampled":
        report["reason"] = "one or more qweight/qzeros/scales payload prefixes could not be sampled"
    return report


def write_gptq_payload_probe_report(
    model_name: str,
    out_dir: Path,
    projection_name: str = "q_proj",
    sample_bytes: int = 64,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gptq_payload_probe.json"
    path.write_text(
        json.dumps(
            build_gptq_payload_probe_report(
                model_name,
                projection_name=projection_name,
                sample_bytes=sample_bytes,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_gptq_checkpoint_metadata_report(model_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gptq_checkpoint_metadata.json"
    path.write_text(json.dumps(inspect_gptq_checkpoint_metadata(model_name), indent=2), encoding="utf-8")
    return path


def write_gptq_checkpoint_source_preflight_report(model_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gptq_checkpoint_source_preflight.json"
    path.write_text(json.dumps(build_gptq_checkpoint_source_preflight_report(model_name), indent=2), encoding="utf-8")
    return path
