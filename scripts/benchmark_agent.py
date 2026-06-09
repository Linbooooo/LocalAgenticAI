from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_agent.agent import LocalAgent
from local_agent.config import AgentConfig


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    files: dict[str, str]
    verify: str


CASES = [
    Case(
        name="create-and-run",
        prompt="Create hello.py that prints exactly Hello, World! and run it to verify the output.",
        files={},
        verify="""python3 - <<'PY'
import subprocess
assert subprocess.check_output(["python3", "hello.py"], text=True).strip() == "Hello, World!"
PY""",
    ),
    Case(
        name="repair-existing-code",
        prompt="Fix clamp.py so clamp(value, low, high) correctly bounds the value. Run useful tests.",
        files={
            "clamp.py": (
                "def clamp(value, low, high):\n"
                "    return max(high, min(low, value))\n"
            )
        },
        verify="""python3 - <<'PY'
from clamp import clamp
assert clamp(5, 0, 10) == 5
assert clamp(-2, 0, 10) == 0
assert clamp(20, 0, 10) == 10
PY""",
    ),
    Case(
        name="implement-algorithm",
        prompt=(
            "Create merge_intervals.py with a merge_intervals(intervals) function. "
            "Handle unsorted, nested, touching, single, and empty inputs. Run tests."
        ),
        files={},
        verify="""python3 - <<'PY'
from merge_intervals import merge_intervals
assert merge_intervals([]) == []
assert merge_intervals([[1, 3]]) == [[1, 3]]
assert merge_intervals([[8, 10], [1, 3], [2, 6], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]
PY""",
    ),
    Case(
        name="extend-without-regression",
        prompt=(
            "Edit stats.py to add median(values). Preserve mean(values), reject an empty median input "
            "with ValueError, and run tests."
        ),
        files={
            "stats.py": (
                "def mean(values):\n"
                "    if not values:\n"
                "        raise ValueError('values must not be empty')\n"
                "    return sum(values) / len(values)\n"
            )
        },
        verify="""python3 - <<'PY'
from stats import mean, median
assert mean([2, 4, 6]) == 4
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
try:
    median([])
except ValueError:
    pass
else:
    raise AssertionError("median([]) must raise ValueError")
PY""",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small hidden-test benchmark against the local coding agent.")
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--limit", type=int, default=len(CASES))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--append-markdown", type=Path)
    args = parser.parse_args()

    selected = [case for case in CASES if not args.case or case.name in args.case][: args.limit]
    results = [run_case(case, args) for case in selected]
    summary = summarize(results, args.model)

    if args.json:
        print(json.dumps({"summary": summary, "cases": results}, indent=2))
    else:
        print_report(results, summary)
    if args.append_markdown:
        append_markdown_row(args.append_markdown, markdown_row(summary))
    return 0 if summary["passed"] == summary["cases"] else 1


def run_case(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(tempfile.mkdtemp(prefix=f"local-agent-{case.name}-"))
    for relative, content in case.files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    config = AgentConfig(
        model=args.model,
        ollama_url=args.url,
        workspace=workspace,
        trust="auto",
        max_steps=args.max_steps,
    )
    config.finalize()
    result = LocalAgent(config).run(case.prompt)
    verification = subprocess.run(
        case.verify,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        executable="/bin/bash",
        timeout=120,
    )
    model_metrics = result.model_metrics
    return {
        "name": case.name,
        "passed": verification.returncode == 0,
        "workspace": str(workspace),
        "turns": result.turns,
        "commands": result.commands,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "tfs_ms": _round(result.time_to_first_shell_ms),
        "ttft_ms_median": _median([metric.ttft_ms for metric in model_metrics]),
        "generation_tps_median": _median([metric.generation_tps for metric in model_metrics]),
        "verification_stdout": verification.stdout.strip(),
        "verification_stderr": verification.stderr.strip(),
        "answer": result.content,
    }


def summarize(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "cases": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "accuracy_percent": round(100 * sum(bool(result["passed"]) for result in results) / len(results), 2),
        "tfs_ms_median": _median([result["tfs_ms"] for result in results if result["tfs_ms"] is not None]),
        "ttft_ms_median": _median([result["ttft_ms_median"] for result in results]),
        "generation_tps_median": _median([result["generation_tps_median"] for result in results]),
        "task_ms_median": _median([result["elapsed_ms"] for result in results]),
    }


def print_report(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status:4} {result['name']}: turns={result['turns']} commands={result['commands']} "
            f"TFS={result['tfs_ms']}ms total={result['elapsed_ms']}ms"
        )
        if not result["passed"]:
            print(f"  workspace: {result['workspace']}")
            print(f"  verifier: {result['verification_stderr'] or result['verification_stdout']}")
            print(f"  answer: {result['answer']}")
    print()
    print(f"Accuracy: {summary['passed']}/{summary['cases']} ({summary['accuracy_percent']}%)")
    print(f"Median TFS: {summary['tfs_ms_median']} ms")
    print(f"Median model TTFT: {summary['ttft_ms_median']} ms")
    print(f"Median generation TPS: {summary['generation_tps_median']}")
    print(f"Median task time: {summary['task_ms_median']} ms")


def markdown_row(summary: dict[str, Any]) -> str:
    return (
        f"| {summary['timestamp_utc'][:10]} | `{summary['model']}` | "
        f"{summary['passed']}/{summary['cases']} | {summary['accuracy_percent']:.2f} | "
        f"{summary['tfs_ms_median']:.2f} | {summary['ttft_ms_median']:.2f} | "
        f"{summary['generation_tps_median']:.2f} | {summary['task_ms_median']:.2f} |"
    )


def append_markdown_row(path: Path, row: str) -> None:
    marker = "<!-- agent-benchmark-results-end -->"
    content = path.read_text(encoding="utf-8")
    if marker not in content:
        raise SystemExit(f"Missing marker in {path}: {marker}")
    path.write_text(content.replace(marker, f"{row}\n{marker}", 1), encoding="utf-8")


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _round(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
