# Operations Guide

This guide covers installation, startup, GPU verification, Docker deployment, and recovery after a reboot.

## Choose One Ollama Deployment

The agent always calls Ollama over local HTTP. Use either:

1. A host/WSL Ollama process at `http://127.0.0.1:11434`.
2. The repository Docker Compose Ollama service.

Do not start both on the same host port. If port `11434` is already allocated, check which Ollama deployment is running before starting another.

## Local WSL Or Linux Setup

Install the Python CLI:

```bash
cd ~/LocalAgenticAI
bash scripts/install_local.sh
. .venv/bin/activate
```

Install and configure Ollama:

```bash
bash scripts/setup_ollama_linux.sh
```

Start Ollama in a dedicated terminal:

```bash
bash scripts/run_ollama_tuned.sh
```

In another terminal:

```bash
cd ~/LocalAgenticAI
. .venv/bin/activate
bash scripts/pull_model.sh
local-agent doctor
local-agent chat
```

The tuned server script enables local-only mode, a 4096-token server context, flash attention, an `q8_0` KV cache, one loaded model, and one parallel request by default.

## Docker Compose Setup

Requirements:

- Docker Desktop or Docker Engine is running.
- Docker Compose is available.
- NVIDIA GPU support is configured if GPU inference is expected.

Start Ollama and pull the model:

```bash
cd ~/LocalAgenticAI
docker compose up -d ollama
docker compose --profile setup run --rm model-pull
```

Check the containerized agent:

```bash
docker compose --profile agent run --rm agent doctor
```

Start chat:

```bash
docker compose --profile agent run --rm agent chat
```

The host can also run the Python CLI against the Compose Ollama service because the service binds to `127.0.0.1:11434`.

## Startup After Reboot

For Docker Desktop on Windows:

1. Start Docker Desktop.
2. Wait until the Linux engine is ready.
3. Open the Ubuntu WSL terminal.
4. Start and verify the stack:

```bash
cd ~/LocalAgenticAI
docker compose up -d ollama
python3 -m local_agent doctor
python3 -m local_agent chat
```

If `docker` is unavailable inside WSL, enable Docker Desktop integration for the Ubuntu distribution under Docker Desktop settings. As a temporary diagnostic, the Windows CLI is usually located at:

```text
/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe
```

For a native WSL/Linux Ollama installation:

```bash
cd ~/LocalAgenticAI
bash scripts/run_ollama_tuned.sh
```

Then open a second terminal and run:

```bash
cd ~/LocalAgenticAI
python3 -m local_agent doctor
python3 -m local_agent chat
```

## Health And GPU Checks

Agent health:

```bash
python3 -m local_agent doctor
```

Ollama API:

```bash
curl http://127.0.0.1:11434/api/version
curl http://127.0.0.1:11434/api/ps
```

Native Ollama:

```bash
ollama list
ollama ps
```

Docker Ollama:

```bash
docker compose exec ollama ollama list
docker compose exec ollama ollama ps
```

GPU visibility:

```bash
nvidia-smi
```

`doctor` and `ollama ps` should report nonzero VRAM use when the model is GPU-backed. Use `python3 -m local_agent preload` to load and keep the configured model resident.

## Common Problems

### Ollama is not reachable

Verify the server:

```bash
curl http://127.0.0.1:11434/api/version
```

If it fails, start the selected Ollama deployment. Do not pull a model until the server is reachable.

### `ollama: command not found`

Either install Ollama with `scripts/setup_ollama_linux.sh` or use the Docker Compose deployment. A Docker container does not automatically install the `ollama` CLI into WSL.

### `docker: command not found` in WSL

Start Docker Desktop and enable WSL integration for Ubuntu. The repository can only use Compose from WSL when the Docker CLI and engine are exposed there.

### Port `11434` is already allocated

Check for an existing container or process:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
ss -ltnp | grep 11434
```

Reuse the healthy existing Ollama service or stop the unintended duplicate. Avoid running multiple Compose projects that publish the same host port.

### Model is installed but not loaded

Loading is demand-driven. Run:

```bash
python3 -m local_agent preload
```

or send the first chat request. The first inference may take longer while the model loads.

### Inference is CPU-bound

Check `python3 -m local_agent doctor`, `ollama ps`, and `nvidia-smi`. In Docker, confirm the Ollama container was created with GPU support and the NVIDIA runtime is available.

## Data Persistence

The Compose deployment stores models in the named volume `local-agentic-ai-ollama-data` by default. Recreating the container does not delete that volume.

To reuse another existing volume:

```bash
OLLAMA_DATA_VOLUME=your_existing_volume \
OLLAMA_DATA_VOLUME_EXTERNAL=true \
docker compose up -d ollama
```

Do not remove the volume unless model deletion is intentional.
