from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContractConstraint:
    kind: str
    first: str
    second: str
    description: str


@dataclass(frozen=True)
class ContractObligation:
    id: str
    kind: str
    description: str
    evidence: tuple[str, ...]
    required: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContract:
    intent: str
    obligations: tuple[ContractObligation, ...] = ()
    constraints: tuple[ContractConstraint, ...] = ()

    def has_obligation(self, kind: str) -> bool:
        return any(obligation.kind == kind for obligation in self.obligations)


@dataclass(frozen=True)
class EvidenceLedger:
    successful_changes: tuple[dict[str, Any], ...] = ()
    successful_reads: tuple[dict[str, Any], ...] = ()
    successful_discovery: tuple[dict[str, Any], ...] = ()
    successful_runs: tuple[dict[str, Any], ...] = ()
    latest_change_step: int = 0

    @property
    def successful_runs_after_latest_change(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            observation
            for observation in self.successful_runs
            if int(observation.get("step", 0)) > self.latest_change_step
        )


def derive_task_contract(
    *,
    task: str,
    intent: str,
    requires_run: bool,
    requires_tests: bool,
    requires_output: bool,
    target_path: str | None = None,
    operation: str | None = None,
) -> TaskContract:
    text = task.lower()
    obligations: list[ContractObligation] = []
    constraints: list[ContractConstraint] = []

    if intent == "edit" and (operation or _looks_like_change_request(text)):
        obligations.append(
            ContractObligation(
                id="workspace_change",
                kind="workspace_change",
                description="Complete the requested workspace file change.",
                evidence=("successful write_file or replace_in_file action",),
                params={"target_path": target_path, "operation": operation},
            )
        )

    if _looks_like_discovery_request(text):
        obligations.append(
            ContractObligation(
                id="workspace_discovery",
                kind="workspace_discovery",
                description="Inspect the workspace to discover the requested files or code before acting on them.",
                evidence=("successful list_files, search_text, or read_file action",),
            )
        )

    if _looks_like_source_display_request(text):
        obligations.append(
            ContractObligation(
                id="source_inspection",
                kind="source_inspection",
                description="Read the requested source or file contents so they can be reported to the user.",
                evidence=("successful read_file action with file content",),
                params={"target_path": target_path},
            )
        )
        obligations.append(
            ContractObligation(
                id="source_report",
                kind="source_report",
                description="Include the observed source or file contents in the final response.",
                evidence=("read_file observation included in final response",),
                params={"target_path": target_path},
            )
        )

    run_count = _requested_run_count(text)
    if requires_run or run_count > 0:
        obligations.append(
            ContractObligation(
                id="local_execution",
                kind="local_execution",
                description="Run the requested local command or program successfully.",
                evidence=("successful run_shell action",),
                params={"min_successes": max(1, run_count)},
            )
        )

    if requires_tests:
        obligations.append(
            ContractObligation(
                id="test_evidence",
                kind="test_evidence",
                description="Observe passing test evidence from a local command.",
                evidence=("successful run_shell action with passing test output",),
            )
        )

    if requires_output:
        obligations.append(
            ContractObligation(
                id="visible_output",
                kind="visible_output",
                description="Produce visible command output or results for the user.",
                evidence=("successful run_shell action with stdout or stderr",),
            )
        )

    if _needs_source_before_execution(text) and _has_kind(obligations, "source_inspection") and _has_kind(
        obligations, "local_execution"
    ):
        constraints.append(
            ContractConstraint(
                kind="before",
                first="source_inspection",
                second="local_execution",
                description="Read/display source before the final successful execution.",
            )
        )

    return TaskContract(intent=intent, obligations=tuple(_dedupe_obligations(obligations)), constraints=tuple(constraints))


def build_evidence_ledger(observations: list[dict[str, Any]]) -> EvidenceLedger:
    successful_changes: list[dict[str, Any]] = []
    successful_reads: list[dict[str, Any]] = []
    successful_discovery: list[dict[str, Any]] = []
    successful_runs: list[dict[str, Any]] = []
    latest_change_step = 0

    for observation in observations:
        action = observation.get("action", {})
        result = observation.get("result", {})
        action_name = action.get("action")
        if not result.get("ok"):
            continue
        step = int(observation.get("step", 0))
        if action_name in {"write_file", "replace_in_file"}:
            successful_changes.append(observation)
            latest_change_step = max(latest_change_step, step)
        elif action_name == "read_file":
            successful_reads.append(observation)
            successful_discovery.append(observation)
        elif action_name in {"list_files", "search_text"}:
            successful_discovery.append(observation)
        elif action_name == "run_shell" and _is_executed_shell_observation(observation):
            successful_runs.append(observation)

    return EvidenceLedger(
        successful_changes=tuple(successful_changes),
        successful_reads=tuple(successful_reads),
        successful_discovery=tuple(successful_discovery),
        successful_runs=tuple(successful_runs),
        latest_change_step=latest_change_step,
    )


def contract_missing(contract: TaskContract, observations: list[dict[str, Any]]) -> list[str]:
    ledger = build_evidence_ledger(observations)
    missing: list[str] = []

    for obligation in contract.obligations:
        if not obligation.required:
            continue
        if obligation.kind == "workspace_change" and not ledger.successful_changes:
            missing.append("The requested workspace change has not been completed.")
        elif obligation.kind == "workspace_discovery" and not ledger.successful_discovery:
            missing.append("The workspace has not been inspected to discover the requested files or code.")
        elif obligation.kind in {"source_inspection", "source_report"} and not ledger.successful_reads:
            missing.append("The requested source or file contents have not been read yet.")
        elif obligation.kind == "local_execution":
            min_successes = int(obligation.params.get("min_successes", 1))
            runs = _relevant_successful_runs(ledger)
            if len(runs) < min_successes:
                if min_successes == 1:
                    missing.append("No successful local command has run for the requested execution.")
                else:
                    missing.append(
                        f"Only {len(runs)} of {min_successes} requested successful local executions have been observed."
                    )
        elif obligation.kind == "test_evidence":
            if not any(_has_passing_test_evidence(observation) for observation in _relevant_successful_runs(ledger)):
                missing.append("Tests were requested, but no passing test evidence has been observed.")
        elif obligation.kind == "visible_output":
            if not any(_has_meaningful_shell_output(observation) for observation in _relevant_successful_runs(ledger)):
                missing.append("Visible command output/results were requested, but none have been observed.")

    for constraint in contract.constraints:
        if constraint.kind == "before" and constraint.first == "source_inspection" and constraint.second == "local_execution":
            read_steps = [int(observation.get("step", 0)) for observation in ledger.successful_reads]
            run_steps = [int(observation.get("step", 0)) for observation in _relevant_successful_runs(ledger)]
            if read_steps and run_steps and min(read_steps) >= max(run_steps):
                missing.append("The source/file contents must be read before the final successful execution.")

    return _dedupe_strings(missing)


def format_task_contract(contract: TaskContract) -> str:
    if not contract.obligations and not contract.constraints:
        return "Task contract: no extra structured obligations inferred."

    lines = ["Task contract:"]
    for obligation in contract.obligations:
        required = "required" if obligation.required else "optional"
        lines.append(f"- {obligation.id} ({obligation.kind}, {required}): {obligation.description}")
        if obligation.evidence:
            lines.append(f"  Evidence: {'; '.join(obligation.evidence)}")
        if obligation.params:
            rendered = ", ".join(f"{key}={value!r}" for key, value in obligation.params.items() if value)
            if rendered:
                lines.append(f"  Params: {rendered}")
    if contract.constraints:
        lines.append("Task constraints:")
        for constraint in contract.constraints:
            lines.append(f"- {constraint.description}")
    return "\n".join(lines)


def format_contract_missing(missing: list[str]) -> str:
    if not missing:
        return ""
    lines = ["Unmet task contract obligations:"]
    lines.extend(f"- {item}" for item in missing)
    lines.append("Choose another local action that produces the missing evidence before using finish.")
    return "\n".join(lines)


def format_contract_evidence_for_final(contract: TaskContract, observations: list[dict[str, Any]]) -> str:
    if not contract.has_obligation("source_report"):
        return ""

    read_observations = [
        observation
        for observation in observations
        if observation.get("action", {}).get("action") == "read_file" and observation.get("result", {}).get("ok")
    ]
    if not read_observations:
        return ""

    lines = ["Observed file contents:"]
    for observation in read_observations[:6]:
        result = observation.get("result", {})
        path = result.get("path", "file")
        content = str(result.get("content", "")).strip()
        if not content:
            lines.append(f"`{path}` is empty.")
            continue
        lines.append(f"`{path}`:")
        lines.append("```text")
        lines.append(content)
        lines.append("```")
    if len(read_observations) > 6:
        lines.append(f"... {len(read_observations) - 6} additional read files omitted from the final response.")
    return "\n".join(lines)


def _looks_like_change_request(text: str) -> bool:
    return bool(re.search(r"\b(write|create|add|modify|update|fix|debug|repair|refactor|implement|generate|build)\b", text))


def _looks_like_discovery_request(text: str) -> bool:
    has_discovery_verb = bool(re.search(r"\b(scan|find|search|look for|list|locate|inspect)\b", text))
    has_workspace_target = bool(
        re.search(r"\b(directory|folder|workspace|repo|repository|project|files?|code|scripts?)\b", text)
        or ".py" in text
        or "python" in text
    )
    return has_discovery_verb and has_workspace_target


def _looks_like_source_display_request(text: str) -> bool:
    has_display_verb = bool(re.search(r"\b(display|show|read|open|view|include)\b", text))
    has_source_target = bool(re.search(r"\b(source|code|contents?|files?|scripts?)\b", text) or ".py" in text)
    output_only = bool(re.search(r"\b(display|show|print)\s+(?:the\s+)?(?:output|results?)\b", text))
    return has_display_verb and has_source_target and not output_only


def _needs_source_before_execution(text: str) -> bool:
    return bool(
        re.search(r"\b(display|show|read|view)\b.*\b(?:then|before)\b.*\b(run|execute)\b", text)
        or re.search(r"\b(run|execute)\b.*\bafter\b.*\b(display|show|read|view)\b", text)
    )


def _requested_run_count(text: str) -> int:
    if not re.search(r"\b(run|execute|rerun)\b", text):
        return 0
    patterns = [
        r"\b(?:run|execute|rerun)\b.*?\b(\d{1,3})\s+times?\b",
        r"\b(\d{1,3})\s+times?\b.*?\b(?:run|execute|rerun)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return max(0, int(match.group(1)))
    word_counts = {
        "once": 1,
        "twice": 2,
        "thrice": 3,
    }
    for word, count in word_counts.items():
        if re.search(rf"\b{word}\b", text):
            return count
    return 0


def _has_kind(obligations: list[ContractObligation], kind: str) -> bool:
    return any(obligation.kind == kind for obligation in obligations)


def _dedupe_obligations(obligations: list[ContractObligation]) -> list[ContractObligation]:
    seen: set[str] = set()
    unique: list[ContractObligation] = []
    for obligation in obligations:
        if obligation.id in seen:
            continue
        seen.add(obligation.id)
        unique.append(obligation)
    return unique


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _relevant_successful_runs(ledger: EvidenceLedger) -> tuple[dict[str, Any], ...]:
    runs = ledger.successful_runs_after_latest_change
    return runs if ledger.latest_change_step else ledger.successful_runs


def _is_executed_shell_observation(observation: dict[str, Any]) -> bool:
    result = observation.get("result", {})
    return any(key in result for key in {"returncode", "stdout", "stderr"})


def _has_passing_test_evidence(observation: dict[str, Any]) -> bool:
    output = _shell_output_text(observation)
    command = str(observation.get("action", {}).get("command", "")).lower()
    if _has_failing_test_output(output):
        return False
    return bool(
        "unittest" in command
        or "pytest" in command
        or re.search(r"\bran\s+\d+\s+tests?\b", output, re.IGNORECASE)
        or re.search(r"\b\d+\s+passed\b", output, re.IGNORECASE)
        or re.search(r"\ball\s+tests?\s+passed\b", output, re.IGNORECASE)
        or re.search(r"\bok\b", output, re.IGNORECASE)
    )


def _has_failing_test_output(output: str) -> bool:
    return bool(
        re.search(r"^FAILED\s*\(", output, re.MULTILINE)
        or re.search(r"^ERROR:", output, re.MULTILINE)
        or re.search(r"^FAIL:", output, re.MULTILINE)
        or "Traceback (most recent call last)" in output
        or "AssertionError" in output
        or "SyntaxError:" in output
    )


def _has_meaningful_shell_output(observation: dict[str, Any]) -> bool:
    return bool(_shell_output_text(observation).strip())


def _shell_output_text(observation: dict[str, Any]) -> str:
    result = observation.get("result", {})
    return f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
