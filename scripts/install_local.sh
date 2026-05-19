#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

echo "Installed local-agent into .venv"
echo "Run it with:"
echo "  . .venv/bin/activate"
echo "  local-agent doctor"

