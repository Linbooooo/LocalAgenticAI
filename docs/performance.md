# Performance

Speed and accuracy are measured separately.

## Metrics

- **TTFT:** milliseconds from request start to the first streamed model text.
- **Generation TPS:** Ollama output tokens divided by generation duration.
- **Prompt TPS:** Ollama input tokens divided by prompt-evaluation duration.
- **TFS:** time to first shell action. This project-specific agent metric includes model planning before the first command.
- **Task latency:** end-to-end time until the agent returns a final answer.
- **Accuracy:** hidden tests or the official SWE-bench evaluator.

## Model Throughput

```bash
make benchmark-model
make benchmark-gpu
make benchmark-cpu
```

The fixed prompt benchmark uses streaming, one warmup, and measured-run medians. `--expected-processor` fails when Ollama reports a different CPU/GPU allocation. `mixed` means only part of the model is in VRAM.

| Date | Label | Processor | Model | Runs | Median TTFT ms | Median generation tok/s | Median prompt tok/s | Median total ms | Context | Max output | Ollama |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-06 | gpu-baseline | gpu | `qwen2.5-coder:14b` | 5 | 126.39 | 73.28 | 3791.98 | 1948.63 | 4096 | 128 | 0.23.2 |
| 2026-06-06 | cpu-baseline | cpu | `qwen2.5-coder:14b` | 5 | 336.35 | 4.90 | 230.49 | 26536.05 | 4096 | 128 | 0.23.2 |
| 2026-06-09 | mixed-gpu-baseline | mixed | `qwen2.5-coder:14b` | 3 | 141.98 | 35.13 | 1771.13 | 2005.79 | 4096 | 64 | 0.23.2 |
<!-- benchmark-results-end -->

The June 9 run used 64 output tokens, so compare it only with runs using the same limit.

Append a comparable row:

```bash
python3 scripts/benchmark_ollama.py \
  --runs 5 \
  --warmup 1 \
  --label mixed-gpu \
  --append-markdown docs/performance.md
```

## Agent Benchmark

```bash
python3 scripts/benchmark_agent.py \
  --append-markdown docs/performance.md
```

| Date | Model | Passed | Accuracy % | Median TFS ms | Median TTFT ms | Median generation tok/s | Median task ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-06-09 | `qwen2.5-coder:14b` | 4/4 | 100.00 | 3643.94 | 476.47 | 28.20 | 33566.95 |
<!-- agent-benchmark-results-end -->

## Switching CPU And GPU

Docker GPU:

```bash
make compose-gpu
python3 -m local_agent preload
make benchmark-gpu
```

Docker CPU:

```bash
make compose-cpu
python3 -m local_agent preload
make benchmark-cpu
```

Native Ollama:

```bash
bash scripts/run_ollama_tuned.sh gpu
bash scripts/run_ollama_tuned.sh cpu
```

Use the same model digest, quantization, prompt, context, output limit, warmup, and concurrency. Record accuracy alongside speed because faster generation does not imply better coding.
