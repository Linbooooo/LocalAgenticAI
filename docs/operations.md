# Operations

## Local WSL Or Linux

```bash
cd ~/LocalAgenticAI
bash scripts/install_local.sh
. .venv/bin/activate
bash scripts/setup_ollama_linux.sh
bash scripts/run_ollama_tuned.sh gpu
```

Keep the Ollama terminal open. In a second terminal:

```bash
cd ~/LocalAgenticAI
. .venv/bin/activate
bash scripts/pull_model.sh
local-agent doctor
local-agent chat
```

## Docker

```bash
cp .env.example .env
make compose-gpu
docker compose --profile setup run --rm model-pull
docker compose --profile agent run --rm agent doctor
docker compose --profile agent run --rm agent chat
```

Use `make compose-cpu` to force CPU inference. Both modes preserve the named Ollama model volume.

## After A Reboot

1. Start Docker Desktop or the native Ollama service.
2. Start the selected Compose mode if using Docker.
3. Verify and open chat:

```bash
cd ~/LocalAgenticAI
make compose-gpu
python3 -m local_agent doctor
python3 -m local_agent chat
```

## Health Checks

```bash
python3 -m local_agent doctor
nvidia-smi
docker compose exec ollama ollama ps
```

`doctor` reports visible CPU, RAM, GPU, Ollama, model installation, and current model residency. A model may be installed but absent from `ps` until the first request or:

```bash
python3 -m local_agent preload
```

## Common Problems

**Ollama unreachable:** start either native Ollama or Compose, not both on port `11434`.

**`ollama` command missing in WSL:** use the Compose service or run `scripts/setup_ollama_linux.sh`.

**Docker unavailable in WSL:** start Docker Desktop and enable Ubuntu WSL integration.

**Model is slow:** check `ollama ps` and `nvidia-smi`; partial VRAM residency is reported as `mixed`.

**Agent can access too much:** local execution inherits the current user's permissions. Run the agent in Docker or another sandbox for untrusted tasks.

## Persistence

Compose stores models in `local-agentic-ai-ollama-data` by default. To reuse another volume, set:

```text
OLLAMA_DATA_VOLUME=your_existing_volume
OLLAMA_DATA_VOLUME_EXTERNAL=true
```

in the ignored `.env` file before starting either CPU or GPU mode.
