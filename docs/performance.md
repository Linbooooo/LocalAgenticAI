# Performance Benchmarks

This document defines a repeatable local benchmark for comparing CPU and GPU inference and tracking performance after future changes.

## Metrics

- **TTFT (time to first token):** client-observed milliseconds from sending the HTTP request until the first non-empty streamed response token arrives.
- **Generation TPS:** output tokens per second, calculated from Ollama's `eval_count / eval_duration`.
- **Prompt TPS:** input tokens per second, calculated from `prompt_eval_count / prompt_eval_duration`.
- **Total latency:** Ollama's total request duration.
- **Load time:** Ollama's model load duration. Warm benchmark runs should make this small; cold-start load time should be measured separately.

Ollama reports durations in nanoseconds. See the official [Ollama usage metrics documentation](https://docs.ollama.com/api/usage).

## Standard Benchmark

The repository benchmark uses:

```text
model: qwen2.5-coder:14b
context: 4096
maximum output: 128 tokens
temperature: 0
warmups: 1
measured runs: 5
prompt: Write a concise Python function that performs binary search on a sorted list, then state its time and space complexity.
```

Run the benchmark:

```bash
make benchmark
```

Assert that the intended processor is active:

```bash
make benchmark-gpu
make benchmark-cpu
```

The script queries `/api/ps` after generation. A run fails when `--expected-processor` does not match the observed VRAM allocation.

## Switching Docker Inference

Both modes use the same Ollama model volume and port. Switching recreates only the Ollama container.

GPU:

```bash
make compose-gpu
python3 -m local_agent preload
make benchmark-gpu
```

CPU:

```bash
make compose-cpu
python3 -m local_agent preload
make benchmark-cpu
```

`docker-compose.yml` is the CPU-safe base and sets `CUDA_VISIBLE_DEVICES=-1`. `docker-compose.gpu.yml` adds the NVIDIA GPU reservation and selects GPU `0` by default. Override the selected GPU with:

```bash
OLLAMA_CUDA_VISIBLE_DEVICES=1 make compose-gpu
```

The default volume is `local-agentic-ai-ollama-data`. To reuse an existing Ollama volume in both modes, pass the same overrides to each command:

```bash
cp .env.example .env
# Set OLLAMA_DATA_VOLUME=your_existing_volume in .env.
# Set OLLAMA_DATA_VOLUME_EXTERNAL=true in .env.
make compose-gpu
make compose-cpu
```

`.env` is ignored by Git, so machine-specific storage and GPU selections remain local.

Ollama documents `CUDA_VISIBLE_DEVICES=-1` as a way to force CPU inference. See [Ollama hardware support](https://docs.ollama.com/gpu) and [Ollama Docker deployment](https://docs.ollama.com/docker).

## Switching Native Ollama

Stop the current `ollama serve` process before changing modes.

GPU:

```bash
bash scripts/run_ollama_tuned.sh gpu
```

CPU:

```bash
bash scripts/run_ollama_tuned.sh cpu
```

Automatic device discovery:

```bash
bash scripts/run_ollama_tuned.sh auto
```

Open another terminal to preload and benchmark the server.

## Current Results

Use the same model, prompt, context, output limit, warmup count, and measured run count when adding rows.

| Date | Label | Processor | Model | Runs | Median TTFT ms | Median generation tok/s | Median prompt tok/s | Median total ms | Context | Max output | Ollama |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-06 | gpu-baseline | gpu | `qwen2.5-coder:14b` | 5 | 126.39 | 73.28 | 3791.98 | 1948.63 | 4096 | 128 | 0.23.2 |
| 2026-06-06 | cpu-baseline | cpu | `qwen2.5-coder:14b` | 5 | 336.35 | 4.90 | 230.49 | 26536.05 | 4096 | 128 | 0.23.2 |
<!-- benchmark-results-end -->

On this machine, the GPU delivered about 15.0x the generation throughput and 13.6x lower total latency than CPU. Its warm TTFT was about 2.7x lower. These ratios describe this model and benchmark configuration on this hardware; record new rows rather than replacing these baselines.

Record a future result automatically:

```bash
python3 scripts/benchmark_ollama.py \
  --runs 5 \
  --warmup 1 \
  --expected-processor cpu \
  --label cpu-baseline \
  --append-markdown docs/performance.md
```

The script inserts the row immediately before the `benchmark-results-end` marker.

## Fair Comparison Rules

1. Use the same model digest and quantization.
2. Keep `num_ctx`, `num_predict`, prompt, temperature, and concurrency unchanged.
3. Use one warmup and compare measured-run medians.
4. Ensure no other model or heavy GPU/CPU workload is active.
5. Record Ollama version and processor detection.
6. Compare quality separately; speed metrics do not detect worse answers.
7. Treat cold model load time separately from warm TTFT.

For future changes, add a labeled row such as `gpu-contract-v2` or `cpu-context-2048` and explain the changed variable in the commit or nearby notes.
