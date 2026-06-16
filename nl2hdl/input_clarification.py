from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import AgentConfig


KNOWN_QUANTIZATION = {"int8_static", "int4_gptq"}
KNOWN_PRUNING = {"none", "magnitude_unstructured"}
KNOWN_DESIGN = {
    "style": {"layer_fsm", "llm_decoder_streaming"},
    "compute_style": {"scalar_fsm", "simd_vector_mac", "systolic_array", "tiled_pe_array", "time_multiplexed_pe"},
    "execution_style": {
        "layer_by_layer",
        "operator_by_operator",
        "token_streaming",
        "llm_decoder_streaming",
        "prefill_decode_split",
        "batch_pipeline",
    },
    "memory_style": {
        "onchip_weight_storage",
        "external_ddr_streaming",
        "external_ddr_gptq_packed",
        "double_buffered_bram",
        "uram_bram_tiled",
    },
    "control_style": {"layer_fsm", "hierarchical_fsm", "top_fsm", "microcoded_controller"},
}
VAGUE_MARKERS = {
    "?",
    "auto",
    "automatic",
    "best",
    "custom",
    "decide",
    "figure out",
    "maybe",
    "tbd",
    "unknown",
    "whatever",
    "알아서",
    "모름",
    "미정",
    "자유롭게",
    "적당히",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _contains_vague_marker(*values: Any) -> bool:
    text = " ".join(_text(value).lower() for value in values)
    return any(marker in text for marker in VAGUE_MARKERS)


def _has_method_context(brief: str | None, candidates: list[Any], extra_options: dict[str, Any]) -> bool:
    return bool(_text(brief) or candidates or extra_options)


def _question(
    *,
    question_id: str,
    scope: str,
    field: str,
    question: str,
    why_needed: str,
    severity: str = "blocking",
) -> dict[str, str]:
    return {
        "id": question_id,
        "scope": scope,
        "field": field,
        "severity": severity,
        "question": question,
        "why_needed": why_needed,
    }


def build_input_clarification_report(config: AgentConfig) -> dict[str, Any]:
    """Build parent-agent clarification questions before HDL packet dispatch."""

    questions: list[dict[str, str]] = []
    optimization_has_context = _has_method_context(
        config.optimization.optimization_brief,
        config.optimization.optimization_candidates,
        config.optimization.extra_options,
    )
    design_has_context = _has_method_context(
        config.design.architecture_brief,
        config.design.design_candidates,
        config.design.extra_options,
    )

    if config.optimization.quantization not in KNOWN_QUANTIZATION and not optimization_has_context:
        questions.append(
            _question(
                question_id="clarify_quantization_method",
                scope="optimization",
                field="optimization.quantization",
                question=(
                    "Please specify the quantization method enough for hardware planning: weight bit width, "
                    "activation precision, scale/zero-point format, calibration source, and numeric tolerance."
                ),
                why_needed=(
                    "The parent cannot choose parser, unpack/dequant, accumulator, and verification packets from "
                    "a free-form quantization label alone."
                ),
            )
        )
    if config.optimization.pruning not in KNOWN_PRUNING and not optimization_has_context:
        questions.append(
            _question(
                question_id="clarify_pruning_or_sparsity_method",
                scope="optimization",
                field="optimization.pruning",
                question=(
                    "Please specify pruning/sparsity granularity, sparse metadata layout, whether zeros are static "
                    "or runtime-skipped, and the accuracy/resource acceptance criteria."
                ),
                why_needed=(
                    "Sparse and pruned methods change module boundaries, memory format, verification vectors, "
                    "and whether dense kernels are still valid."
                ),
            )
        )
    if _contains_vague_marker(
        config.optimization.quantization,
        config.optimization.pruning,
        config.optimization.optimization_brief,
    ) and not config.optimization.optimization_candidates:
        questions.append(
            _question(
                question_id="clarify_vague_optimization_intent",
                scope="optimization",
                field="optimization",
                question=(
                    "The optimization intent looks open-ended. Please list one to three concrete optimization "
                    "candidates, with priority and what evidence should decide between them."
                ),
                why_needed=(
                    "The parent can compare candidates, but it should not silently invent the optimization method "
                    "before assigning HDL agents."
                ),
            )
        )

    unclear_design_fields = [
        field
        for field, known_values in KNOWN_DESIGN.items()
        if _text(getattr(config.design, field)) not in known_values
    ]
    if unclear_design_fields and not design_has_context:
        questions.append(
            _question(
                question_id="clarify_hardware_design_methodology",
                scope="design",
                field="design",
                question=(
                    "Please explain the hardware design methodology enough for module planning: compute fabric, "
                    "execution schedule, memory movement, control structure, and any preferred tradeoff."
                ),
                why_needed=(
                    "The parent cannot safely decompose the accelerator into minimal HDL packets from unknown "
                    f"design fields alone: {', '.join(unclear_design_fields)}."
                ),
            )
        )
    if _contains_vague_marker(
        config.design.style,
        config.design.compute_style,
        config.design.execution_style,
        config.design.memory_style,
        config.design.control_style,
        config.design.architecture_brief,
    ) and not config.design.design_candidates:
        questions.append(
            _question(
                question_id="clarify_vague_architecture_intent",
                scope="design",
                field="design.architecture_brief",
                question=(
                    "The architecture intent looks open-ended. Please list candidate hardware design directions "
                    "or state whether the parent may propose candidates before dispatch."
                ),
                why_needed=(
                    "Free-form architecture guidance is allowed, but unclear guidance should become explicit "
                    "questions instead of hidden assumptions."
                ),
            )
        )

    return {
        "artifact": "input_clarification_questions",
        "status": "needs_clarification" if questions else "clear",
        "requires_user_response": bool(questions),
        "question_count": len(questions),
        "questions": questions,
        "checked_fields": {
            "optimization": asdict(config.optimization),
            "design": asdict(config.design),
        },
        "parent_policy": (
            "If requires_user_response is true, stop before HDL sub-agent dispatch and ask these questions. "
            "If false, preserve the free-form fields in manifests and continue planning."
        ),
    }
