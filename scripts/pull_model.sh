#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${LOCAL_AGENT_MODEL:-qwen3.5:9b-q4_K_M}}"
ollama pull "${MODEL}"
