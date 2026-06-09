# Local Agentic AI

A small, local-only coding agent powered by Ollama. The model can inspect a repository, edit files, run commands, observe failures, and continue until it can answer.

The implementation follows the same minimal principle popularized by [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent): one linear conversation and one general tool, Bash.

## Architecture

```text
user request
    -> local Ollama model
    -> fenced Bash command
    -> guarded subprocess in the workspace
    -> stdout/stderr/exit code appended to history
    -> repeat or return a plain-text answer
```

There is no intent classifier, RAG pipeline, task-contract engine, skills router, or framework dependency. The model plans; the harness supplies execution, observations, context limits, confirmation, and metrics.

Main files:

- `local_agent/agent.py`: the complete agent loop and Bash-block protocol.
- `local_agent/tools.py`: subprocess execution, confirmation, and basic safety policy.
- `local_agent/ollama_client.py`: local streaming Ollama client and inference metrics.
- `local_agent/context.py`: bounded conversation packing.
- `local_agent/config.py`: model, context, runtime, and trust settings.
- `scripts/benchmark_agent.py`: hidden-test coding benchmark with TFS, TTFT, and TPS.
- `scripts/swebench.py`: SWE-bench Lite patch generation and official evaluator handoff.

See [Architecture](docs/architecture.md), [Evaluation](docs/evaluation.md), [Performance](docs/performance.md), and [Operations](docs/operations.md).

## Model Choice

The default is `qwen2.5-coder:14b`. It is a practical balance for this machine's RTX 3080 Ti with 12 GB VRAM and 16 GB system RAM: substantially more capable than small 7B coders, while remaining usable with partial GPU offload and a 4096-token context.

The model is configurable. Larger models generally improve planning and code quality but require more memory and reduce throughput. Smaller models are faster but fail more often on multi-step repair.

## Quick Start

Requirements: Python 3.11+, Git, and a local Ollama server.

```bash
bash scripts/install_local.sh
. .venv/bin/activate
bash scripts/run_ollama_tuned.sh gpu
```

In another terminal:

```bash
bash scripts/pull_model.sh
python3 -m local_agent doctor
python3 -m local_agent chat
```

Example:

```text
> Fix the failing parser tests, run them, and summarize the change.
Allow shell command:
sed -n '1,240p' parser.py
[y/N] y
...
```

Use `--yes` for non-interactive benchmark or trusted sandbox runs:

```bash
python3 -m local_agent --yes "Create hello.py and run it."
```

## Docker

Build the agent:

```bash
docker build -t local-agentic-ai:latest .
```

Start Ollama on GPU and run the containerized CLI:

```bash
cp .env.example .env
make compose-gpu
docker compose --profile setup run --rm model-pull
docker compose --profile agent run --rm agent chat
```

Switch to CPU with:

```bash
make compose-cpu
```

Both modes reuse the same model volume. The base Compose file disables CUDA; `docker-compose.gpu.yml` adds NVIDIA GPU access.

## Configuration

Common environment variables:

```text
LOCAL_AGENT_MODEL=qwen2.5-coder:14b
OLLAMA_URL=http://127.0.0.1:11434
LOCAL_AGENT_WORKSPACE=.
LOCAL_AGENT_NUM_CTX=4096
LOCAL_AGENT_NUM_PREDICT=1024
LOCAL_AGENT_MAX_STEPS=20
LOCAL_AGENT_SHELL_TIMEOUT=120
LOCAL_AGENT_TRUST=ask
LOCAL_AGENT_ALLOW_NETWORK_TOOLS=false
```

The context packer reserves `num_predict` tokens for output and keeps the newest complete messages that fit. Exact repository state is recovered by inspecting files again rather than trusting an old summary.

## Test And Benchmark

```bash
make test
make benchmark-model
make benchmark-agent
```

- `benchmark-model` measures TTFT, prompt TPS, generation TPS, and total latency.
- `benchmark-agent` uses hidden tests to measure coding accuracy, time to first shell action (TFS), and end-to-end task time.
- `make swebench` generates a patch for one SWE-bench Lite task after the optional benchmark packages are installed.

The official SWE-bench Docker evaluator is the source of truth for resolved-task accuracy. See [Evaluation](docs/evaluation.md).

## Safety Boundary

Commands run as independent subprocesses from the configured workspace. Mutating commands require confirmation unless `--yes` is used. Common network and destructive commands are blocked by default.

This is not an OS sandbox: a local shell inherits the current user's filesystem permissions. Use the Docker deployment or another sandbox when running untrusted prompts or repositories.
