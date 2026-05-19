# Local Agentic AI

Local Agentic AI is a private, local-only coding agent for this machine. It talks to an Ollama server bound to loopback, uses a local model, and can inspect, edit, and run commands inside a workspace with explicit safety gates.

## Model Choice

Default model: `qwen2.5-coder:14b`

Why this default fits this machine:

- Your RTX 3080 Ti has 12 GB VRAM. Ollama lists `qwen2.5-coder:14b` at 9.0 GB with a 32K context window, leaving room for KV cache and GPU overhead.
- `qwen3-coder:30b` is more agent-trained, but Ollama lists it at 19 GB. That is a poor fit for 12 GB VRAM and 16 GB WSL memory.
- The 14B Qwen coder model is Apache-2.0 and code-specific, so it is the best default balance of local speed, quality, and fit.

Useful references:

- Qwen2.5 Coder in Ollama: https://ollama.com/library/qwen2.5-coder
- Qwen3 Coder in Ollama: https://ollama.com/library/qwen3-coder
- Ollama local-only/cloud controls and GPU memory notes: https://docs.ollama.com/faq

## Quick Start

Install the CLI locally:

```bash
bash scripts/install_local.sh
```

Install Ollama if needed:

```bash
bash scripts/setup_ollama_linux.sh
```

Start Ollama with local-only, GPU-friendly settings:

```bash
bash scripts/run_ollama_tuned.sh
```

In another terminal, pull the selected model:

```bash
ollama pull qwen2.5-coder:14b
```

Run a health check:

```bash
python3 -m local_agent doctor
```

Ask the agent to work in the current workspace:

```bash
python3 -m local_agent "Inspect this repository and summarize what it does."
```

Start an interactive session:

```bash
python3 -m local_agent chat
```

Allow the agent to execute write/shell tools without prompting:

```bash
python3 -m local_agent --yes "Add tests for the parser and run them."
```

Or, after `scripts/install_local.sh`:

```bash
local-agent chat
```

## Docker Deployment

Build the agent image:

```bash
docker build -t local-agentic-ai:latest .
```

Run against Ollama on the host:

```bash
docker run --rm -it \
  --network host \
  -e OLLAMA_URL=http://127.0.0.1:11434 \
  -v "$PWD:/workspace" \
  local-agentic-ai:latest chat
```

Run the full local stack with Docker Compose:

```bash
docker compose up -d ollama
docker compose --profile setup run --rm model-pull
docker compose --profile agent run --rm agent chat
```

The Compose stack keeps Ollama on Docker's local network and stores downloaded models in the `ollama-data` volume.

To reuse an existing Ollama Docker volume:

```bash
OLLAMA_DATA_VOLUME=your_existing_volume OLLAMA_DATA_VOLUME_EXTERNAL=true docker compose up -d ollama
```

## Architecture

This project does not depend on OpenClaw. It uses a small local agent loop in [local_agent/agent.py](local_agent/agent.py), an Ollama client in [local_agent/ollama_client.py](local_agent/ollama_client.py), and workspace tools in [local_agent/tools.py](local_agent/tools.py). Keeping the loop in-repo makes the local-only policy, file confinement, and tool behavior easy to audit.

OpenClaw can still be used next to this project as a separate UI or orchestrator, but it is not required for install or deployment.

## Context Management

The agent does not rely on Ollama to blindly truncate long chats. Before each model call it builds a bounded context pack:

- Always keep the main system/local-only instructions.
- Keep the newest request and recent tool loop messages.
- Condense older messages into a short system summary.
- Summarize old tool results by outcome and metadata instead of replaying large outputs.
- Re-read files with tools when exact source content matters.

## Safety Model

- The model endpoint must be loopback: `127.0.0.1`, `localhost`, or `::1`.
- Container deployments may also use the local Docker hostnames `ollama` and `host.docker.internal`.
- Cloud Ollama features are disabled in the tuned runner with `OLLAMA_NO_CLOUD=1`.
- Filesystem tools are confined to the configured workspace.
- Network-capable shell commands such as `curl`, `wget`, `ssh`, `pip install`, and `npm install` are blocked by default.
- Mutating tools prompt before running unless `--yes` is supplied.

The default config lives in [local_agent/config.py](local_agent/config.py). You can override settings with a JSON file:

```bash
python3 -m local_agent --config local-agent.json chat
```

Example:

```json
{
  "model": "qwen2.5-coder:14b",
  "ollama_url": "http://127.0.0.1:11434",
  "workspace": ".",
  "num_ctx": 4096,
  "max_num_ctx": 4096,
  "min_num_ctx": 2048,
  "num_predict": 2048,
  "temperature": 0.2,
  "top_p": 0.9,
  "keep_alive": "30m",
  "ollama_timeout": 300,
  "trust": "ask",
  "allow_network_tools": false
}
```

Environment variables are also supported:

- `LOCAL_AGENT_MODEL`
- `OLLAMA_URL`
- `LOCAL_AGENT_WORKSPACE`
- `LOCAL_AGENT_NUM_CTX`
- `LOCAL_AGENT_MAX_NUM_CTX`
- `LOCAL_AGENT_MIN_NUM_CTX`
- `LOCAL_AGENT_NUM_PREDICT`
- `LOCAL_AGENT_TRUST`
- `LOCAL_AGENT_ALLOW_NETWORK_TOOLS`
- `LOCAL_AGENT_OLLAMA_TIMEOUT`

## Upgrade Path

This repository defaults to `qwen2.5-coder:14b` at a conservative 4K context because that loads reliably on 16 GB system memory while still using the RTX 3080 Ti. It also caps context with `LOCAL_AGENT_MAX_NUM_CTX` because Ollama can sometimes accept an oversized context and then become extremely slow instead of failing fast. If Ollama does reject a request because the requested context is too large for available memory, the agent automatically halves `num_ctx` and retries down to `LOCAL_AGENT_MIN_NUM_CTX` instead of crashing.

If you later give WSL/Ollama 32 GB or more RAM, raise both `LOCAL_AGENT_NUM_CTX` and `LOCAL_AGENT_MAX_NUM_CTX` to `8192` or `16384`, or try `qwen3-coder:30b` for stronger agentic coding. If you add a 24 GB GPU, Qwen3 Coder becomes the obvious default.

## GPU Notes

The agent container does not have to see the GPU directly when Ollama runs as a separate service. The important question is whether the Ollama service is using GPU. `local-agent doctor` prints Ollama's loaded model state, including `size_vram`. If that value is `0`, the model is likely CPU-bound and will be much slower.

For GPU-first deployment on Linux/WSL, use the Compose `ollama` service, which is configured with `gpus: all`, NVIDIA runtime environment variables, and a loopback port binding on `127.0.0.1:11434`. A host Ollama app can also work, but it must be configured separately to use the GPU.
