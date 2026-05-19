# Local Agentic AI

Local Agentic AI is a local-only coding agent backed by Ollama. It can inspect a workspace, answer questions about code, make scoped edits, and run local verification commands while keeping model inference and tool execution on the host machine.

The project is intentionally small: a Python CLI, an Ollama client, a deterministic tool-policy layer, workspace-confined tools, and Docker deployment files.

## Features

- Local Ollama inference with no required cloud API.
- Interactive chat and one-shot task execution.
- Workspace-confined file tools.
- Intent-gated tool exposure: chat requests get no tools, inspection requests get read-only tools, and edit tools are exposed only for explicit edit requests.
- Confirmation prompts for mutating tools unless `--yes` is supplied.
- Bounded context packing for longer sessions.
- Docker and Docker Compose support, including GPU-enabled Ollama deployment.

## Default Model

Default model: `qwen2.5-coder:14b`

This model is a practical default for local coding work on 12 GB-class GPUs. It is small enough to run locally with a conservative context window while still being useful for code inspection and edits. Larger models may work better on machines with more VRAM and system memory.

Useful references:

- Qwen2.5 Coder in Ollama: https://ollama.com/library/qwen2.5-coder
- Ollama documentation: https://docs.ollama.com

## Requirements

- Python 3.11+
- Ollama
- Git
- Docker and Docker Compose for containerized deployment
- NVIDIA Container Toolkit for GPU-backed Docker inference

## Local Installation

Install the CLI in a local virtual environment:

```bash
bash scripts/install_local.sh
```

Install Ollama if needed:

```bash
bash scripts/setup_ollama_linux.sh
```

Start Ollama with local-first settings:

```bash
bash scripts/run_ollama_tuned.sh
```

Pull the default model:

```bash
ollama pull qwen2.5-coder:14b
```

Run a health check:

```bash
python3 -m local_agent doctor
```

Start an interactive session:

```bash
python3 -m local_agent chat
```

Run a one-shot task:

```bash
python3 -m local_agent "Inspect this repository and summarize the project."
```

After `scripts/install_local.sh`, the console entry point is also available:

```bash
local-agent chat
```

## Docker Deployment

Build the agent image:

```bash
docker build -t local-agentic-ai:latest .
```

Run the agent against an Ollama server on the host:

```bash
docker run --rm -it \
  --network host \
  -e OLLAMA_URL=http://127.0.0.1:11434 \
  -v "$PWD:/workspace" \
  local-agentic-ai:latest chat
```

Run the full stack with Docker Compose:

```bash
docker compose up -d ollama
docker compose --profile setup run --rm model-pull
docker compose --profile agent run --rm agent chat
```

Reuse an existing Ollama Docker volume:

```bash
OLLAMA_DATA_VOLUME=your_existing_volume OLLAMA_DATA_VOLUME_EXTERNAL=true docker compose up -d ollama
```

The Compose Ollama service binds to `127.0.0.1:11434`, disables Ollama cloud mode, and is configured for NVIDIA GPU access with `gpus: all`.

## Configuration

The default configuration lives in [local_agent/config.py](local_agent/config.py). Override it with JSON:

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

Supported environment variables:

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

## Architecture

Request flow:

```text
user request
-> deterministic tool policy
-> selected tool schemas
-> Ollama chat request
-> tool execution, if allowed
-> final response
```

Main components:

- [local_agent/agent.py](local_agent/agent.py): model/tool loop and adaptive context retry.
- [local_agent/tool_policy.py](local_agent/tool_policy.py): deterministic request classification and allowed tool sets.
- [local_agent/tools.py](local_agent/tools.py): workspace file tools, shell tool, hardware profile tool, and enforcement.
- [local_agent/context.py](local_agent/context.py): bounded context packing.
- [local_agent/ollama_client.py](local_agent/ollama_client.py): local Ollama HTTP client.

Tool policies:

- `chat`: no tools.
- `read`: `list_files`, `read_file`, `search_text`.
- `hardware`: `hardware_profile`.
- `shell`: read tools, hardware tool, and `run_shell`.
- `edit`: all tools, with mutating tools still confirmation-gated.

## Safety

- Ollama endpoints must be local: loopback, `host.docker.internal`, or the Compose service name `ollama`.
- Filesystem operations are confined to the configured workspace.
- Network-capable shell commands are blocked by default.
- Destructive shell patterns are blocked by default.
- Mutating tools prompt before execution unless `trust` is set to `auto` or `--yes` is supplied.
- Tool availability is determined before the model call, so casual chat does not expose file or shell tools.

## Context And Performance

The default context is conservative for local hardware:

- `num_ctx`: requested context window.
- `max_num_ctx`: hard cap used to avoid accidentally loading an oversized context.
- `min_num_ctx`: lower bound used for automatic retry after memory errors.

Before each model call, older conversation is compacted into a short summary while the current request and recent tool loop remain intact. Exact source details should be re-read with file tools instead of relying on stale context.

Use `local-agent doctor` to inspect the active Ollama server, loaded models, context length, and reported VRAM usage. If `size_vram` is `0`, inference is likely CPU-bound.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Build the container:

```bash
docker build -t local-agentic-ai:latest .
```
