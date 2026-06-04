from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class CodingSkill:
    name: str
    purpose: str
    instructions: tuple[str, ...]


CODING_CHANGE_SKILL = CodingSkill(
    name="coding-change",
    purpose="Make scoped code changes and verify real results.",
    instructions=(
        "Read the target file or nearby project context before editing existing code.",
        "Preserve existing style, public APIs, and project structure unless the user asks to change them.",
        "When the user asks to implement and test, create or update the code, run a relevant check, then finish from observed output.",
        "Do not finish by only describing code when the task asks you to write, run, or test it.",
    ),
)

PROJECT_DISCOVERY_SKILL = CodingSkill(
    name="project-discovery",
    purpose="Find the project layout before choosing commands or imports.",
    instructions=(
        "Use workspace files such as pyproject.toml, README.md, Makefile, package manifests, and tests to infer layout.",
        "Prefer package-root commands over direct file execution when imports depend on the repository root.",
        "Use search_text for named functions, classes, files, or error strings instead of guessing locations.",
    ),
)

PYTHON_TESTING_SKILL = CodingSkill(
    name="python-testing",
    purpose="Run Python code and tests from the correct project context.",
    instructions=(
        "Prefer python3 over python.",
        "Run unittest files under tests/test_*.py with python3 -m unittest discover -s tests -p <filename> from the workspace root.",
        "If a test file import fails only when direct-running tests/test_*.py, rerun with unittest discovery before editing imports.",
        "Use pytest only when project files show pytest is available or already used.",
    ),
)

DEBUGGING_SKILL = CodingSkill(
    name="debugging",
    purpose="Turn failed runs into a concrete next fix.",
    instructions=(
        "Treat command output and tracebacks as ground truth.",
        "Classify the failure before editing: import/path problem, runtime exception, assertion failure, command/tool problem, or bad expected test data.",
        "For assertion failures, compare the failing input, expected value, and actual value against the problem statement or an independent oracle.",
        "Do not change expected values just to match current output; fix the implementation unless the expected value is independently proven wrong.",
        "Fix one likely root cause, then rerun the narrowest relevant command.",
        "Do not claim success until a verification command succeeds.",
    ),
)

ALGORITHM_VERIFICATION_SKILL = CodingSkill(
    name="algorithm-verification",
    purpose="Check coding-problem solutions with meaningful tests.",
    instructions=(
        "For algorithm tasks, include edge cases such as duplicates, empty inputs, singletons, no-solution cases, and boundary values when relevant.",
        "When feasible, create a small brute-force oracle or hand-verified expected values instead of guessing expected output.",
        "For returned-index problems, verify expected indices are valid, distinct when required, and point to values satisfying the target condition.",
        "For two-sum-style returned-index tasks, prefer a validity-check helper or brute-force expected result over arbitrary fixed index lists.",
        "For generated tests, keep expected values stable during repair unless an independent oracle proves the test is wrong.",
        "For order-insensitive outputs, normalize before comparing.",
        "If multiple answers are valid, test validity properties rather than one arbitrary ordering.",
    ),
)


def select_coding_skills(
    task: str,
    intent: str,
    workspace_files: list[str],
    observations: list[dict[str, Any]] | None = None,
) -> list[CodingSkill]:
    if intent not in {"read", "edit", "shell"}:
        return []

    text = task.lower()
    files = [path.replace("\\", "/").lower() for path in workspace_files]
    selected: list[CodingSkill] = []

    if intent == "edit" or _looks_like_coding_request(text):
        selected.append(CODING_CHANGE_SKILL)

    if _has_project_layout(files) or _mentions_existing_code(text):
        selected.append(PROJECT_DISCOVERY_SKILL)

    if _is_python_context(text, files):
        selected.append(PYTHON_TESTING_SKILL)

    if _mentions_failure(text) or _has_failed_observation(observations or []):
        selected.append(DEBUGGING_SKILL)

    if _is_algorithm_task(text):
        selected.append(ALGORITHM_VERIFICATION_SKILL)

    return _dedupe_skills(selected)


def format_coding_skills(skills: list[CodingSkill]) -> str:
    if not skills:
        return ""

    lines = ["Active coding skills:"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.purpose}")
        for instruction in skill.instructions:
            lines.append(f"  - {instruction}")
    return "\n".join(lines)


def _dedupe_skills(skills: list[CodingSkill]) -> list[CodingSkill]:
    seen: set[str] = set()
    unique: list[CodingSkill] = []
    for skill in skills:
        if skill.name in seen:
            continue
        seen.add(skill.name)
        unique.append(skill)
    return unique


def _looks_like_coding_request(text: str) -> bool:
    markers = {
        "code",
        "function",
        "class",
        "script",
        "test",
        "debug",
        "fix",
        "refactor",
        "implement",
        "compile",
        "build",
        "import",
    }
    return any(marker in text for marker in markers)


def _mentions_existing_code(text: str) -> bool:
    return any(marker in text for marker in {".py", "file", "repo", "project", "module", "package", "tests/"})


def _has_project_layout(files: list[str]) -> bool:
    names = {PurePosixPath(path).name for path in files}
    return bool({"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "makefile", "package.json"} & names)


def _is_python_context(text: str, files: list[str]) -> bool:
    if "python" in text or ".py" in text or "unittest" in text or "pytest" in text:
        return True
    return any(path.endswith(".py") or path == "pyproject.toml" or path.startswith("tests/") for path in files)


def _mentions_failure(text: str) -> bool:
    markers = {
        "traceback",
        "exception",
        "error",
        "failing",
        "failed",
        "fails",
        "debug",
        "fix",
        "modulenotfounderror",
        "assertionerror",
    }
    return any(marker in text for marker in markers)


def _has_failed_observation(observations: list[dict[str, Any]]) -> bool:
    return any(not observation.get("result", {}).get("ok", True) for observation in observations)


def _is_algorithm_task(text: str) -> bool:
    normalized = text.replace("_", " ").replace("-", " ")
    markers = {
        "algorithm",
        "leetcode",
        "two sum",
        "combination sum",
        "binary search",
        "dynamic programming",
        "graph",
        "tree",
        "array",
        "linked list",
    }
    return any(marker in normalized for marker in markers)
