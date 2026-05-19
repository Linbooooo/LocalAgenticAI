#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${LOCAL_AGENT_MODEL:-qwen2.5-coder:14b}}"
ollama pull "${MODEL}"

