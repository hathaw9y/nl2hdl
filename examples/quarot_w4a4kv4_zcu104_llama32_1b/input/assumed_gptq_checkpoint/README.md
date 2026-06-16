# Assumed GPTQ Checkpoint Fixture

This directory is a sparse local fixture for the QuaRot W4A4KV4 example.

It exists so `nl2hdl agent --mode inspect` can exercise the parent planning,
GPTQ metadata, payload-prefix probe, and sub-agent prompt generation flow
without downloading or storing the real LLaMA-3.2-1B checkpoint.

The generated `model.safetensors` file is not real model data. It contains a
synthetic safetensors header and sparse zero-filled tensor regions with the
target LLaMA projection shapes. It proves path/schema handling only.

Do not use this fixture to claim:

- numeric GPTQ correctness;
- full checkpoint tensor materialization;
- real LLaMA logits;
- board-level or model-level target execution.
