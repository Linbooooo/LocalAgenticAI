from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


SMOKE_PROMPTS = [
    "write a python file that prints out Hello World",
    "write a python file that prints out Hello World and run it",
    "create a python script named greet_user.py that prints hi linbo and run it",
    "write a python file that implements two sum and test it and display the results",
    "write a python file that solves valid parentheses and test it. Display the results.",
    "write a python file that implements binary search and test it and display output",
    "write a python file that outputs the nth fibonacci number given an input n. test it and display the results.",
    "write a python file that solves 3sum and test it. Display the results.",
    "write a python file that checks whether a string is a palindrome and test it. Display results.",
]

MEDIUM_PROMPTS = [
    "write a python file that implements merge sort and include test cases. Run it and show results.",
    "write a python file that implements merge intervals. Include tests for overlapping, touching, nested, unsorted, single interval, and empty inputs. Run it and display results.",
    "write a python file that implements top k frequent words. Break frequency ties lexicographically. Include tests for ties, k=1, k larger than unique words, and repeated words. Run it and display results.",
    "write a python file that implements a MinStack with push, pop, top, and get_min in O(1). Include tests with duplicate minimums, negative values, and pop behavior. Run it and display results.",
    "write a python file that solves decode string such as 3[a2[c]]. Include tests for nesting, multi-digit repeat counts, adjacent encoded groups, and plain text. Run it and display results.",
    "write a python file that determines if all courses can be finished from prerequisites. Include tests for acyclic graphs, simple cycles, disconnected graphs, and self-cycles. Run it and display results.",
    "write a python file that implements an LRU cache with capacity eviction. Include tests for updates, get refreshing recency, capacity one, missing keys, and eviction order. Run it and display results.",
]

HARD_PROMPTS = [
    "write a python file that solves combination sum ii where candidates may contain duplicates and each number may be used once. Include tests for duplicate candidates, no solution, empty result, and order-insensitive comparison. Run it and display results.",
    "write a python file that computes edit distance between two strings. Include tests for empty strings, identical strings, insertion, deletion, substitution, and a classic horse to ros case. Run it and display results.",
    "write a python file that solves word ladder shortest transformation length using BFS. Include tests for reachable path, unreachable path, begin equals end, and multiple shortest choices. Run it and display results.",
    "write a python file that implements regular expression matching with dot and star only, matching the entire string. Include tests for a*, dot, empty string, false partial matches, and nested star cases. Run it and display results.",
    "write a python file that counts N queens solutions. Include tests for n=1, n=2, n=3, n=4, and n=5. Run it and display results.",
    "write a python file that evaluates arithmetic expressions with plus, minus, parentheses, and spaces. Include tests for nested parentheses, unary negative numbers, whitespace, and multi-digit numbers. Run it and display results.",
]

PROMPT_SUITES = {
    "smoke": SMOKE_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "hard": HARD_PROMPTS,
    "all": SMOKE_PROMPTS + MEDIUM_PROMPTS + HARD_PROMPTS,
}

FAILURE_MARKERS = [
    "I could not get a valid local action",
    "Stopped after",
    "Could not repair",
    "Could not find a corrective local action",
    "Completion criteria not met",
    "Traceback (most recent call last)",
    "SyntaxError:",
    "AssertionError",
    "NO TESTS RAN",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live local-agent coding smoke prompts.")
    parser.add_argument(
        "--suite",
        choices=sorted(PROMPT_SUITES),
        default="smoke",
        help="Prompt suite to run when --prompt is not provided.",
    )
    parser.add_argument("--prompt", action="append", help="Prompt to run. Can be repeated.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per prompt in seconds.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON results.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    prompts = args.prompt or PROMPT_SUITES[args.suite]
    results = [run_prompt(repo, prompt, args.timeout) for prompt in prompts]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    return 1 if any(result["status"] != "PASS" for result in results) else 0


def run_prompt(repo: Path, prompt: str, timeout: int) -> dict[str, object]:
    workspace = tempfile.mkdtemp(prefix="local-agent-eval-")
    command = [
        "python3",
        "-m",
        "local_agent",
        "--workspace",
        workspace,
        "--yes",
        *prompt.split(),
    ]
    started = time.time()

    try:
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=timeout)
        output = completed.stdout.strip()
        if completed.stderr.strip():
            output = f"{output}\n\nstderr:\n{completed.stderr.strip()}".strip()
        completed_successfully = "Completed and verified successfully." in output
        scored_output = _last_command_section(output) if completed_successfully else output
        failed = completed.returncode != 0 or any(marker in scored_output for marker in FAILURE_MARKERS)
        status = "FAIL" if failed else "PASS"
        return {
            "status": status,
            "seconds": round(time.time() - started, 1),
            "workspace": workspace,
            "prompt": prompt,
            "returncode": completed.returncode,
            "output": output,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "FAIL",
            "seconds": round(time.time() - started, 1),
            "workspace": workspace,
            "prompt": prompt,
            "returncode": None,
            "output": f"Timed out after {timeout}s.\nstdout={exc.stdout}\nstderr={exc.stderr}",
        }


def print_report(results: list[dict[str, object]]) -> None:
    for index, result in enumerate(results, start=1):
        print(f"=== {index:02d} {result['status']} {result['seconds']}s ===")
        print(result["prompt"])
        print(f"workspace: {result['workspace']}")
        print(_tail(str(result["output"])))
        print()
    passed = sum(result["status"] == "PASS" for result in results)
    print(f"SUMMARY {passed}/{len(results)} passed")


def _tail(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _last_command_section(output: str) -> str:
    marker = "\n$ "
    if marker not in output:
        return output
    return output.rsplit(marker, 1)[-1]


if __name__ == "__main__":
    raise SystemExit(main())
