#!/usr/bin/env bash
set -euo pipefail

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama is already installed: $(ollama --version || true)"
else
  echo "Installing Ollama. This downloads the official installer from ollama.com."
  curl -fsSL https://ollama.com/install.sh | sh
fi

mkdir -p "${HOME}/.ollama"
cat > "${HOME}/.ollama/server.json" <<'JSON'
{
  "disable_ollama_cloud": true
}
JSON

echo "Configured Ollama local-only mode in ${HOME}/.ollama/server.json"
echo "Next:"
echo "  bash scripts/run_ollama_tuned.sh"
echo "  ollama pull qwen2.5-coder:14b"

