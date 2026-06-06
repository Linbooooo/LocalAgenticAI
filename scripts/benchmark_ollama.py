from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_PROMPT = (
    "Write a concise Python function that performs binary search on a sorted list, "
    "then state its time and space complexity."
)


@dataclass(frozen=True)
class RunMetrics:
    ttft_ms: float
    total_ms: float
    load_ms: float
    prompt_tokens: int
    prompt_tps: float
    output_tokens: int
    generation_tps: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama model with streaming TTFT and token metrics.")
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="Local Ollama base URL.")
    parser.add_argument("--model", default="qwen2.5-coder:14b", help="Ollama model name.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fixed prompt used for every measured run.")
    parser.add_argument("--runs", type=int, default=5, help="Number of measured runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs excluded from the summary.")
    parser.add_argument("--num-predict", type=int, default=128, help="Maximum generated tokens per run.")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Context window used for the benchmark.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per request in seconds.")
    parser.add_argument("--label", default="", help="Optional label such as gpu-baseline or cpu-baseline.")
    parser.add_argument("--expected-processor", choices=["cpu", "gpu", "mixed"], help="Fail if Ollama reports another mode.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    parser.add_argument("--append-markdown", type=Path, help="Append one result row to an existing Markdown table.")
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be at least 1 and --warmup cannot be negative")

    version = get_json(args.url, "api/version", args.timeout).get("version", "unknown")
    for index in range(args.warmup):
        print(f"Warmup {index + 1}/{args.warmup}...", file=sys.stderr, flush=True)
        benchmark_run(args)

    runs: list[RunMetrics] = []
    for index in range(args.runs):
        print(f"Measured run {index + 1}/{args.runs}...", file=sys.stderr, flush=True)
        run = benchmark_run(args)
        runs.append(run)
        print(
            f"  TTFT={run.ttft_ms:.1f} ms  generation={run.generation_tps:.2f} tok/s  "
            f"prompt={run.prompt_tps:.2f} tok/s  total={run.total_ms:.1f} ms",
            file=sys.stderr,
            flush=True,
        )

    processor = detect_processor(get_json(args.url, "api/ps", args.timeout), args.model)
    if args.expected_processor and processor != args.expected_processor:
        raise SystemExit(f"Expected {args.expected_processor} inference, but Ollama reports {processor}.")

    result = summarize(
        runs,
        model=args.model,
        processor=processor,
        ollama_version=str(version),
        label=args.label,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        prompt=args.prompt,
    )
    row = markdown_row(result)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print_summary(result)
        print()
        print("Markdown row:")
        print(row)

    if args.append_markdown:
        append_markdown_row(args.append_markdown, row)
        print(f"Appended result to {args.append_markdown}", file=sys.stderr)
    return 0


def benchmark_run(args: argparse.Namespace) -> RunMetrics:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
        },
    }
    request = Request(
        urljoin(args.url.rstrip("/") + "/", "api/generate"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    final: dict[str, Any] = {}
    try:
        with urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line)
                if event.get("response") and first_token_at is None:
                    first_token_at = time.perf_counter()
                if event.get("done"):
                    final = event
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama returned HTTP {exc.code}: {body or exc.reason}") from exc
    except URLError as exc:
        raise SystemExit(f"Could not reach Ollama: {exc}") from exc

    finished = time.perf_counter()
    if first_token_at is None:
        raise SystemExit("Ollama completed without emitting a response token.")
    if not final:
        raise SystemExit("Ollama stream ended without a final metrics event.")

    return RunMetrics(
        ttft_ms=(first_token_at - started) * 1000,
        total_ms=_ns_to_ms(final.get("total_duration")) or (finished - started) * 1000,
        load_ms=_ns_to_ms(final.get("load_duration")),
        prompt_tokens=int(final.get("prompt_eval_count", 0) or 0),
        prompt_tps=_tokens_per_second(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
        output_tokens=int(final.get("eval_count", 0) or 0),
        generation_tps=_tokens_per_second(final.get("eval_count"), final.get("eval_duration")),
    )


def summarize(
    runs: list[RunMetrics],
    *,
    model: str,
    processor: str,
    ollama_version: str,
    label: str,
    num_ctx: int,
    num_predict: int,
    prompt: str,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label or f"{processor}-benchmark",
        "model": model,
        "processor": processor,
        "ollama_version": ollama_version,
        "runs": len(runs),
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "prompt": prompt,
        "ttft_ms_median": round(statistics.median(run.ttft_ms for run in runs), 2),
        "ttft_ms_mean": round(statistics.fmean(run.ttft_ms for run in runs), 2),
        "generation_tps_median": round(statistics.median(run.generation_tps for run in runs), 2),
        "generation_tps_mean": round(statistics.fmean(run.generation_tps for run in runs), 2),
        "prompt_tps_median": round(statistics.median(run.prompt_tps for run in runs), 2),
        "total_ms_median": round(statistics.median(run.total_ms for run in runs), 2),
        "load_ms_median": round(statistics.median(run.load_ms for run in runs), 2),
        "output_tokens_median": round(statistics.median(run.output_tokens for run in runs), 2),
    }


def detect_processor(payload: dict[str, Any], model: str) -> str:
    models = payload.get("models", [])
    selected = next(
        (
            item
            for item in models
            if item.get("name") == model or item.get("model") == model or str(item.get("name", "")).startswith(model + ":")
        ),
        models[0] if models else {},
    )
    size = int(selected.get("size", 0) or 0)
    size_vram = int(selected.get("size_vram", 0) or 0)
    if size_vram <= 0:
        return "cpu"
    if size > 0 and size_vram >= size * 0.95:
        return "gpu"
    return "mixed"


def markdown_row(result: dict[str, Any]) -> str:
    date = str(result["timestamp_utc"])[:10]
    return (
        f"| {date} | {result['label']} | {result['processor']} | `{result['model']}` | "
        f"{result['runs']} | {result['ttft_ms_median']:.2f} | {result['generation_tps_median']:.2f} | "
        f"{result['prompt_tps_median']:.2f} | {result['total_ms_median']:.2f} | "
        f"{result['num_ctx']} | {result['num_predict']} | {result['ollama_version']} |"
    )


def append_markdown_row(path: Path, row: str) -> None:
    marker = "<!-- benchmark-results-end -->"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in content:
        updated = content.replace(marker, f"{row}\n{marker}", 1)
    else:
        separator = "" if not content or content.endswith("\n") else "\n"
        updated = f"{content}{separator}{row}\n"
    path.write_text(updated, encoding="utf-8")


def print_summary(result: dict[str, Any]) -> None:
    print(f"Processor: {result['processor']}")
    print(f"Model: {result['model']}")
    print(f"Median TTFT: {result['ttft_ms_median']:.2f} ms")
    print(f"Median generation TPS: {result['generation_tps_median']:.2f} tok/s")
    print(f"Median prompt TPS: {result['prompt_tps_median']:.2f} tok/s")
    print(f"Median total latency: {result['total_ms_median']:.2f} ms")
    print(f"Median model load time: {result['load_ms_median']:.2f} ms")


def get_json(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    request = Request(urljoin(base_url.rstrip("/") + "/", path), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Could not query Ollama {path}: {exc}") from exc


def _tokens_per_second(count: Any, duration_ns: Any) -> float:
    count_value = int(count or 0)
    duration_value = int(duration_ns or 0)
    if count_value <= 0 or duration_value <= 0:
        return 0.0
    return count_value / (duration_value / 1_000_000_000)


def _ns_to_ms(value: Any) -> float:
    return int(value or 0) / 1_000_000


if __name__ == "__main__":
    raise SystemExit(main())
