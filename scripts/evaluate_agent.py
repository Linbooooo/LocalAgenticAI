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

AGENTIC_CASES = [
    {
        "prompt": "write a python file named count_runs.py that prints agentic-count and run it 3 times.",
        "checks": {
            "contains": ["agentic-count"],
            "count_at_least": {"agentic-count": 3},
            "file_exists": ["count_runs.py"],
        },
    },
    {
        "prompt": "write a python file named show_then_run.py that prints source-visible. Display the code, then run it.",
        "checks": {
            "contains": ["Observed file contents:", "show_then_run.py", "print", "source-visible"],
            "count_at_most": {"$ python3 show_then_run.py": 1},
            "file_exists": ["show_then_run.py"],
        },
    },
    {
        "prompt": "write and run a python program named cleanup_target.py that prints cleanup-ok, then delete the file.",
        "checks": {
            "contains": ["cleanup-ok"],
            "file_absent": ["cleanup_target.py"],
        },
    },
    {
        "prompt": (
            "write a python file named update_twice.py that prints original-marker, run it, change original-marker to updated-marker, "
            "run it again, then delete the file."
        ),
        "checks": {
            "contains": ["original-marker", "updated-marker"],
            "file_absent": ["update_twice.py"],
        },
    },
    {
        "prompt": (
            "write and run a simple python program named feeling_hello.py that prints Hello Eval, "
            "and then tell me how you are feeling today."
        ),
        "checks": {
            "contains": ["Hello Eval"],
            "any_contains_ci": [
                [
                    "i am feeling",
                    "i'm feeling",
                    "i feel",
                    "i am doing",
                    "i'm doing",
                    "i do not have feelings",
                    "i don't have feelings",
                ]
            ],
            "not_contains_ci": ["how are you feeling today?", "how about yourself", "what about you"],
            "count_at_most": {"$ python3 feeling_hello.py": 1},
            "file_exists": ["feeling_hello.py"],
        },
    },
]

PROMPT_SUITES["agentic"] = AGENTIC_CASES

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
    prompts = [{"prompt": prompt} for prompt in args.prompt] if args.prompt else PROMPT_SUITES[args.suite]
    results = [run_case(repo, prompt, args.timeout) for prompt in prompts]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    return 1 if any(result["status"] != "PASS" for result in results) else 0


def run_case(repo: Path, case: str | dict[str, object], timeout: int) -> dict[str, object]:
    if isinstance(case, str):
        prompt = case
        checks: dict[str, object] = {}
    else:
        prompt = str(case["prompt"])
        checks = case.get("checks", {})
        if not isinstance(checks, dict):
            checks = {}

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
        check_failures = _check_result(Path(workspace), output, checks)
        failed = failed or bool(check_failures)
        status = "FAIL" if failed else "PASS"
        return {
            "status": status,
            "seconds": round(time.time() - started, 1),
            "workspace": workspace,
            "prompt": prompt,
            "returncode": completed.returncode,
            "check_failures": check_failures,
            "output": output,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "FAIL",
            "seconds": round(time.time() - started, 1),
            "workspace": workspace,
            "prompt": prompt,
            "returncode": None,
            "check_failures": ["Timed out."],
            "output": f"Timed out after {timeout}s.\nstdout={exc.stdout}\nstderr={exc.stderr}",
        }


def print_report(results: list[dict[str, object]]) -> None:
    for index, result in enumerate(results, start=1):
        print(f"=== {index:02d} {result['status']} {result['seconds']}s ===")
        print(result["prompt"])
        print(f"workspace: {result['workspace']}")
        failures = result.get("check_failures") or []
        if failures:
            print("check failures:")
            for failure in failures:
                print(f"- {failure}")
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


def _check_result(workspace: Path, output: str, checks: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for text in _string_list(checks.get("contains")):
        if text not in output:
            failures.append(f"Output missing required text: {text!r}")

    output_lower = output.lower()
    for text in _string_list(checks.get("contains_ci")):
        if text.lower() not in output_lower:
            failures.append(f"Output missing required text, case-insensitive: {text!r}")

    for text in _string_list(checks.get("not_contains")):
        if text in output:
            failures.append(f"Output contains forbidden text: {text!r}")

    for text in _string_list(checks.get("not_contains_ci")):
        if text.lower() in output_lower:
            failures.append(f"Output contains forbidden text, case-insensitive: {text!r}")

    any_contains = checks.get("any_contains")
    if isinstance(any_contains, list):
        for group in any_contains:
            choices = _string_list(group)
            if choices and not any(choice in output for choice in choices):
                failures.append(f"Output missing one of: {choices!r}")

    any_contains_ci = checks.get("any_contains_ci")
    if isinstance(any_contains_ci, list):
        for group in any_contains_ci:
            choices = _string_list(group)
            if choices and not any(choice.lower() in output_lower for choice in choices):
                failures.append(f"Output missing one of, case-insensitive: {choices!r}")

    count_at_least = checks.get("count_at_least")
    if isinstance(count_at_least, dict):
        for text, minimum in count_at_least.items():
            if not isinstance(text, str):
                continue
            try:
                required = int(minimum)
            except (TypeError, ValueError):
                continue
            observed = output.count(text)
            if observed < required:
                failures.append(f"Output contains {text!r} {observed} times, expected at least {required}.")

    count_at_most = checks.get("count_at_most")
    if isinstance(count_at_most, dict):
        for text, maximum in count_at_most.items():
            if not isinstance(text, str):
                continue
            try:
                allowed = int(maximum)
            except (TypeError, ValueError):
                continue
            observed = output.count(text)
            if observed > allowed:
                failures.append(f"Output contains {text!r} {observed} times, expected at most {allowed}.")

    for path in _string_list(checks.get("file_exists")):
        if not (workspace / path).exists():
            failures.append(f"Expected file to exist: {path}")

    for path in _string_list(checks.get("file_absent")):
        if (workspace / path).exists():
            failures.append(f"Expected file to be absent: {path}")

    return failures


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


if __name__ == "__main__":
    raise SystemExit(main())
