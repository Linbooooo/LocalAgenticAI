#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${OLLAMA_DEVICE:-gpu}}"

case "${MODE}" in
  cpu)
    export CUDA_VISIBLE_DEVICES="-1"
    ;;
  gpu)
    export CUDA_VISIBLE_DEVICES="${OLLAMA_CUDA_VISIBLE_DEVICES:-0}"
    ;;
  auto)
    unset CUDA_VISIBLE_DEVICES
    ;;
  *)
    echo "Usage: $0 [cpu|gpu|auto]" >&2
    exit 2
    ;;
esac

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

echo "Starting Ollama on ${OLLAMA_HOST}"
echo "Compute mode: ${MODE}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Local-only cloud mode: ${OLLAMA_NO_CLOUD}"
echo "Context length: ${OLLAMA_CONTEXT_LENGTH}"
echo "Flash attention: ${OLLAMA_FLASH_ATTENTION}"
echo "KV cache: ${OLLAMA_KV_CACHE_TYPE}"

exec ollama serve
