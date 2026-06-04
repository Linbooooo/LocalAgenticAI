# Local Agentic AI

Local Agentic AI is a local-only coding agent backed by Ollama. It can inspect a workspace, answer questions about code, make scoped edits, and run local verification commands while keeping model inference and tool execution on the host machine.

The project is intentionally small: a Python CLI, an Ollama client, a deterministic control layer implemented by `LocalAgent`, workspace-confined actions, and Docker deployment files.

## Features

- Local Ollama inference with no required cloud API.
- Interactive chat and one-shot task execution.
- Workspace-confined file actions.
- Direct shell command execution for requests like `execute "nvidia-smi"`.
- Iterative action loop for multi-step work such as editing, running tests, and responding with real output.
- Model-based semantic routing for chat, read, hardware, shell, and edit tasks.
- Isolated route/action protocol calls with explicit current request, workspace snapshot, and structured agent state.
- Deterministic action validation for repeated failed commands, missing test directories, missing stdout entry points, and premature success claims.
- Lightweight coding skills for project discovery, Python testing, debugging, and algorithm verification.
- Confirmation prompts for mutating actions unless `--yes` is supplied.
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
  "allow_network_tools": false,
  "max_steps": 24
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
- `LOCAL_AGENT_MAX_STEPS`

## Architecture

See [docs/architecture.md](docs/architecture.md) for a Mermaid flowchart of the agent loop.

Request flow:

```text
user request
-> exact direct command check
-> model-based semantic route validated by LocalAgent
-> isolated action protocol with current request, workspace snapshot, and agent state
-> iterative local action loop when work requires tools
-> Ollama chat request when reasoning only is needed
-> final response
```

Main components:

- [local_agent/agent.py](local_agent/agent.py): `LocalAgent`, which implements the control flow, model calls, direct commands, and iterative action handling.
- [local_agent/skills.py](local_agent/skills.py): compact procedural coding skills selected for relevant action requests.
- [local_agent/tool_policy.py](local_agent/tool_policy.py): exact direct-command extractor.
- [local_agent/tools.py](local_agent/tools.py): workspace file actions, shell execution, hardware profile, and safety checks.
- [local_agent/context.py](local_agent/context.py): bounded context packing.
- [local_agent/ollama_client.py](local_agent/ollama_client.py): local Ollama HTTP client.

`LocalAgent` routes after asking the model for a semantic route:

- `chat`: send the request to the model without workspace actions.
- `read`: let the model inspect local files through read-only actions, then answer.
- `hardware`: gather local hardware/Ollama status, then ask the model.
- `shell`: run an exact command directly, or let the model choose local shell actions when the command is implicit.
- `edit`: let the model edit, run checks, inspect results, and continue until it can give a final answer.

Routing and action-mode responses go through protocol layers before execution. `LocalAgent` requests JSON-mode route/action output from Ollama, validates required fields, downgrades low-confidence action routes to chat, retries malformed action responses, and can salvage an obvious Python code block into a named file write when the user explicitly asked to create that file.

The normal chat path may use conversation history, but route/action protocol calls are isolated from the prior assistant transcript. The action prompt is self-contained: it includes the current user request, a bounded workspace snapshot, selected coding skills, completion requirements, prior observations from the current task, and structured state such as the last written file and last shell command. This prevents stale outputs from an earlier task from poisoning the next tool decision while still letting the model resolve references like "test it" from agent state.

Repair planning is also validated outside the model. The harness rejects duplicate failed shell commands before rerunning them, blocks repeated identical rewrites, keeps command comparisons normalized across common Python invocation forms, and stops when the model cannot produce a corrective action after a failed verification.

## Coding Skills

Skills are not extra tools. They are small instruction packs injected into the action prompt when the task and workspace call for them. The model still chooses actions, and the deterministic tool layer still validates and executes those actions.

Current built-in skills:

- `coding-change`: inspect relevant code, keep edits scoped, and verify before claiming success.
- `project-discovery`: infer layout from files such as `pyproject.toml`, `README.md`, `Makefile`, package manifests, and tests.
- `python-testing`: prefer `python3`, run `tests/test_*.py` through unittest discovery from the workspace root, and avoid import fixes when the issue is direct test-file execution.
- `debugging`: classify failures from real output, compare assertions against the problem or an oracle, fix one likely root cause, and rerun the narrowest relevant command.
- `algorithm-verification`: test algorithmic solutions with edge cases, independent expected values, returned-index validity checks, and normalization when multiple outputs are acceptable.

This keeps common coding workflows close to the agent without widening the tool permission surface.

## Evaluation

Evaluate the agent at three levels:

- Unit tests for the harness: routing validation, skill selection, action protocol repair, shell normalization, stop-on-success behavior, and workspace safety.
- Offline coding tasks: fixed prompts in temporary workspaces with assertions on final files, command results, number of steps, and whether the agent verified its work.
- Live model scorecards: run the same task set after model, prompt, or skill changes and track pass rate, unnecessary tool calls, repeated actions, failed imports, false success claims, and average steps to completion.

Good eval tasks should cover simple file creation, existing-code edits, Python test generation, traceback repair, package import handling, and medium algorithm problems where expected outputs can be independently checked.

## Safety

- Ollama endpoints must be local: loopback, `host.docker.internal`, or the Compose service name `ollama`.
- Filesystem operations are confined to the configured workspace.
- Network-capable shell commands are blocked by default.
- Destructive shell patterns are blocked by default.
- Mutating actions prompt before execution unless `trust` is set to `auto` or `--yes` is supplied.
- The model decides the semantic route; the deterministic `LocalAgent` control layer validates that route and enforces safety boundaries before any tool runs.
- Malformed or incomplete action responses are rejected and retried before any local side effect runs.

## Context And Performance

The default context is conservative for local hardware:

- `num_ctx`: requested context window.
- `max_num_ctx`: hard cap used to avoid accidentally loading an oversized context.
- `min_num_ctx`: lower bound used for automatic retry after memory errors.

Before each model call, older conversation is compacted into a short summary while the current request remains intact. Action-mode requests include a small deterministic workspace snapshot and previous action results instead of vector search.

Use `local-agent doctor` to inspect the active Ollama server, loaded models, context length, and reported VRAM usage. If `size_vram` is `0`, inference is likely CPU-bound.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run live model evaluation prompts in temporary workspaces:

```bash
python3 scripts/evaluate_agent.py --suite smoke
python3 scripts/evaluate_agent.py --suite medium --timeout 300
python3 scripts/evaluate_agent.py --suite hard --timeout 300
```

The Makefile exposes the same flows:

```bash
make test
make eval-smoke
make eval-medium
make eval-hard
```

Live evals are intentionally model-sensitive. A failed live prompt means the current model plus prompt stack did not solve that task reliably; unit tests still cover the deterministic harness guarantees.

Build the container:

```bash
docker build -t local-agentic-ai:latest .
```
