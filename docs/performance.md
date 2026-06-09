# Performance

Speed and accuracy are measured separately.

The current default is `qwen3.5:9b-q4_K_M` with an 8192-token context. Older rows remain as historical baselines for `qwen2.5-coder:14b`.

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
| 2026-06-09 | qwen35-gpu-baseline | gpu | `qwen3.5:9b-q4_K_M` | 5 | 228.53 | 90.62 | 946.88 | 1559.39 | 4096 | 128 | 0.23.2 |
| 2026-06-09 | qwen35-cpu-baseline | cpu | `qwen3.5:9b-q4_K_M` | 5 | 1961.26 | 5.02 | 20.24 | 24950.18 | 4096 | 128 | 0.23.2 |
<!-- benchmark-results-end -->

The June 9 run used 64 output tokens, so compare it only with runs using the same limit.

At the shared 4096-context and 128-output settings, Qwen3.5 generated 23.7% faster
on GPU and reduced median total latency by 20.0% compared with the June 6
Qwen2.5-Coder GPU baseline. Its median GPU TTFT was 80.8% slower. On CPU,
generation improved by 2.4% and total latency fell by 6.0%, but TTFT increased
substantially. CPU mode remains useful for portability, not interactive speed.

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
| 2026-06-09 | `qwen3.5:9b-q4_K_M` | 4/4 | 100.00 | 2970.97 | 350.51 | 90.75 | 6586.24 |
<!-- agent-benchmark-results-end -->

Both models passed the four focused hidden-test cases. Qwen3.5 reduced median
task time by 80.4%, reduced median TFS by 18.5%, and generated 221.8% faster.
The older agent run used its 4096-token project default; the Qwen3.5 run used
the new 8192-token default, so this is an operational comparison rather than a
strict context-controlled model comparison.
This benchmark is intentionally small and should not be interpreted as general
repository-level coding accuracy.

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
