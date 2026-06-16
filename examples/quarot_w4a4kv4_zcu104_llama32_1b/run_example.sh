#!/usr/bin/env bash
set -euo pipefail

python3 -m nl2hdl agent \
  --model meta-llama/Llama-3.2-1B \
  --spec examples/quarot_w4a4kv4_zcu104_llama32_1b/input.yaml \
  --mode inspect \
  --out examples/quarot_w4a4kv4_zcu104_llama32_1b/run/inspect \
  --verbose
