from __future__ import annotations

import re
import shlex
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
    successful_deletes: tuple[dict[str, Any], ...] = ()
    successful_reads: tuple[dict[str, Any], ...] = ()
    successful_discovery: tuple[dict[str, Any], ...] = ()
    successful_runs: tuple[dict[str, Any], ...] = ()
    successful_edit_reviews: tuple[dict[str, Any], ...] = ()
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
                evidence=("successful write_file, replace_in_file, or edit_repair action",),
                params={
                    "target_path": target_path,
                    "operation": operation,
                    "edit_scope": _fallback_edit_scope(text),
                },
            )
        )

    if intent == "edit" and operation in {"update", "repair"}:
        obligations.append(
            ContractObligation(
                id="source_inspection",
                kind="source_inspection",
                description="Inspect the existing target source before changing it.",
                evidence=("successful read_file action before the workspace change",),
                params={"target_path": target_path},
            )
        )
        obligations.append(
            ContractObligation(
                id="edit_review",
                kind="edit_review",
                description="Review the edited source against the user's requested transformation.",
                evidence=("successful edit_review after the latest workspace change",),
                params={"target_path": target_path},
            )
        )
        constraints.extend(
            [
                ContractConstraint(
                    kind="before",
                    first="source_inspection",
                    second="workspace_change",
                    description="Inspect the existing source before changing it.",
                ),
                ContractConstraint(
                    kind="before",
                    first="workspace_change",
                    second="edit_review",
                    description="Review the resulting source after changing it.",
                ),
            ]
        )

    if intent == "edit" and _looks_like_delete_request(text):
        obligations.append(
            ContractObligation(
                id="workspace_delete",
                kind="workspace_delete",
                description="Delete the requested workspace file.",
                evidence=("successful delete_file action",),
                params={"target_path": target_path},
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
                params={"min_successes": max(1, run_count), "target_path": target_path},
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

    if _looks_like_assistant_response_request(task):
        obligations.append(
            ContractObligation(
                id="assistant_response",
                kind="assistant_response",
                description="Provide the requested conversational answer to the user.",
                evidence=("final assistant response addresses the requested conversational answer",),
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


def contract_from_model_json(value: Any, *, fallback: TaskContract, task: str = "") -> TaskContract:
    if not isinstance(value, dict):
        return fallback

    obligations = _filter_model_obligations(
        _parse_model_obligations(value.get("obligations")),
        task,
        fallback,
    )
    constraints = _filter_constraints(
        _parse_model_constraints(value.get("constraints"), {obligation.id for obligation in obligations}),
        obligations,
    )
    if not obligations:
        return fallback

    merged = _merge_contracts(
        TaskContract(intent=fallback.intent, obligations=tuple(obligations), constraints=tuple(constraints)),
        fallback,
    )
    return merged


def build_evidence_ledger(observations: list[dict[str, Any]]) -> EvidenceLedger:
    successful_changes: list[dict[str, Any]] = []
    successful_deletes: list[dict[str, Any]] = []
    successful_reads: list[dict[str, Any]] = []
    successful_discovery: list[dict[str, Any]] = []
    successful_runs: list[dict[str, Any]] = []
    successful_edit_reviews: list[dict[str, Any]] = []
    latest_change_step = 0

    for observation in observations:
        action = observation.get("action", {})
        result = observation.get("result", {})
        action_name = action.get("action")
        if not result.get("ok"):
            continue
        step = int(observation.get("step", 0))
        if action_name in {"write_file", "replace_in_file", "edit_repair"}:
            successful_changes.append(observation)
            latest_change_step = max(latest_change_step, step)
        elif action_name == "delete_file":
            successful_deletes.append(observation)
        elif action_name == "read_file":
            successful_reads.append(observation)
            successful_discovery.append(observation)
        elif action_name in {"list_files", "search_text"}:
            successful_discovery.append(observation)
        elif action_name == "run_shell" and _is_executed_shell_observation(observation):
            successful_runs.append(observation)
        elif action_name == "edit_review":
            successful_edit_reviews.append(observation)

    return EvidenceLedger(
        successful_changes=tuple(successful_changes),
        successful_deletes=tuple(successful_deletes),
        successful_reads=tuple(successful_reads),
        successful_discovery=tuple(successful_discovery),
        successful_runs=tuple(successful_runs),
        successful_edit_reviews=tuple(successful_edit_reviews),
        latest_change_step=latest_change_step,
    )


def contract_missing(contract: TaskContract, observations: list[dict[str, Any]]) -> list[str]:
    ledger = build_evidence_ledger(observations)
    missing: list[str] = []

    for obligation in contract.obligations:
        if not obligation.required:
            continue
        candidate_steps = _candidate_steps(obligation, ledger)
        if obligation.kind == "assistant_response":
            continue
        if obligation.kind == "local_execution":
            min_successes = int(obligation.params.get("min_successes", 1) or 1)
            if len(candidate_steps) < min_successes:
                if min_successes == 1:
                    missing.append(f"{obligation.description} has no successful local execution evidence.")
                else:
                    missing.append(
                        f"Only {len(candidate_steps)} of {min_successes} requested successful local executions have been observed for {obligation.description}."
                    )
        elif not candidate_steps and obligation.kind == "workspace_change":
            missing.append("The requested workspace change has not been completed.")
        elif not candidate_steps and obligation.kind == "workspace_delete":
            missing.append("The requested workspace file has not been deleted.")
        elif not candidate_steps and obligation.kind == "workspace_discovery":
            missing.append("The workspace has not been inspected to discover the requested files or code.")
        elif not candidate_steps and obligation.kind == "source_inspection":
            missing.append("The requested source or file contents have not been read yet.")
        elif not candidate_steps and obligation.kind == "source_report":
            missing.append("The final requested source or file contents have not been read yet.")
        elif not candidate_steps and obligation.kind == "edit_review":
            missing.append("The edited source has not passed review against the requested transformation.")
        elif obligation.kind == "test_evidence":
            if not any(_has_passing_test_evidence(observation) for observation in _relevant_successful_runs(ledger)):
                missing.append("Tests were requested, but no passing test evidence has been observed.")
        elif obligation.kind == "visible_output":
            if not any(_has_meaningful_shell_output(observation) for observation in _relevant_successful_runs(ledger)):
                missing.append("Visible command output/results were requested, but none have been observed.")

    for constraint in contract.constraints:
        if constraint.kind != "before":
            continue
        first = _obligation_by_id(contract, constraint.first)
        second = _obligation_by_id(contract, constraint.second)
        if first is None or second is None:
            continue
        first_steps = _candidate_steps(first, ledger)
        second_steps = _candidate_steps(second, ledger)
        if first_steps and second_steps and not any(first_step < second_step for first_step in first_steps for second_step in second_steps):
            missing.append(constraint.description or f"{constraint.first} must happen before {constraint.second}.")

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

    ledger = build_evidence_ledger(observations)
    read_observations = [
        observation
        for observation in ledger.successful_reads
        if not ledger.latest_change_step or int(observation.get("step", 0)) > ledger.latest_change_step
    ]
    if not read_observations:
        return ""

    latest_by_path: dict[str, dict[str, Any]] = {}
    for observation in read_observations:
        path = str(observation.get("result", {}).get("path", "file"))
        latest_by_path[path] = observation

    lines = ["Observed file contents:"]
    selected = list(latest_by_path.values())
    for observation in selected[:6]:
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
    if len(selected) > 6:
        lines.append(f"... {len(selected) - 6} additional read files omitted from the final response.")
    return "\n".join(lines)


def _looks_like_change_request(text: str) -> bool:
    return bool(re.search(r"\b(write|create|add|modify|update|fix|debug|repair|refactor|implement|generate|build)\b", text))


def _looks_like_delete_request(text: str) -> bool:
    return bool(re.search(r"\b(delete|remove|unlink)\b", text))


def _looks_like_assistant_response_request(task: str) -> bool:
    text = task.lower()
    return bool(
        re.search(r"\b(tell me|explain|summarize|describe|answer|report|how are you|how you are|feeling|feel today)\b", text)
    )


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
    explicit_source_display = bool(
        re.search(r"\b(display|show|read|open|view|include)\b.{0,50}\b(source|code|contents?|files?|scripts?)\b", text)
    )
    if explicit_source_display:
        return True
    output_only = bool(
        re.search(r"\b(display|show|print)\b.{0,50}\b(?:command\s+)?(?:output|results?)\b", text)
    )
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
    verbs = re.findall(r"\b(?:run|execute|rerun)\b", text)
    return len(verbs) if len(verbs) > 1 else 0


def _fallback_edit_scope(text: str) -> str | None:
    test_terms = r"(?:tests?|test cases?|examples?|fixtures?|demo(?: inputs?)?|assertions?)"
    change_terms = r"(?:change|modify|update|replace|add|rewrite|make)"
    if re.search(rf"\b{change_terms}\b.{{0,80}}\b{test_terms}\b", text):
        if not re.search(r"\b(fix|debug|repair|refactor|implementation|algorithm|function|class|production code)\b", text):
            return "tests_only"
    return None


def _has_kind(obligations: list[ContractObligation], kind: str) -> bool:
    return any(obligation.kind == kind for obligation in obligations)


def _parse_model_obligations(value: Any) -> list[ContractObligation]:
    if not isinstance(value, list):
        return []

    obligations: list[ContractObligation] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value[:32], start=1):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        if kind not in _allowed_obligation_kinds():
            continue
        obligation_id = _safe_identifier(str(item.get("id", "")).strip()) or f"{kind}_{index}"
        while obligation_id in seen_ids:
            obligation_id = f"{obligation_id}_{index}"
        seen_ids.add(obligation_id)

        description = _one_line(str(item.get("description", ""))) or kind.replace("_", " ")
        evidence = tuple(
            _one_line(str(entry))
            for entry in item.get("evidence", [])
            if isinstance(entry, str) and _one_line(entry)
        )
        obligations.append(
            ContractObligation(
                id=obligation_id,
                kind=kind,
                description=_truncate(description, 240),
                evidence=evidence,
                required=_bool_value(item.get("required", True)),
                params=_sanitize_params(item.get("params", {})),
            )
        )
    return obligations


def _filter_model_obligations(
    obligations: list[ContractObligation],
    task: str,
    fallback: TaskContract,
) -> list[ContractObligation]:
    filtered: list[ContractObligation] = []
    allowed_execution_count = _fallback_execution_count(fallback)
    model_executions = [obligation for obligation in obligations if obligation.kind == "local_execution"]
    retained_execution_ids = {
        obligation.id
        for obligation in model_executions[-allowed_execution_count:]
    } if allowed_execution_count else set()
    for obligation in obligations:
        if obligation.kind == "assistant_response" and not _looks_like_assistant_response_request(task):
            continue
        if obligation.kind == "source_report" and not fallback.has_obligation("source_report"):
            continue
        if obligation.kind in {"local_execution", "test_evidence", "visible_output"} and not fallback.has_obligation(
            obligation.kind
        ):
            continue
        if obligation.kind == "local_execution" and obligation.id not in retained_execution_ids:
            continue
        filtered.append(obligation)
    return filtered


def _fallback_execution_count(contract: TaskContract) -> int:
    return sum(
        int(obligation.params.get("min_successes", 1) or 1)
        for obligation in contract.obligations
        if obligation.kind == "local_execution" and obligation.required
    )


def _parse_model_constraints(value: Any, obligation_ids: set[str]) -> list[ContractConstraint]:
    if not isinstance(value, list):
        return []

    constraints: list[ContractConstraint] = []
    for item in value[:64]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        first = _safe_identifier(str(item.get("first", "")).strip())
        second = _safe_identifier(str(item.get("second", "")).strip())
        if kind != "before" or first not in obligation_ids or second not in obligation_ids or first == second:
            continue
        description = _one_line(str(item.get("description", ""))) or f"{first} must happen before {second}."
        constraints.append(
            ContractConstraint(
                kind=kind,
                first=first,
                second=second,
                description=_truncate(description, 240),
            )
        )
    return constraints


def _merge_contracts(primary: TaskContract, fallback: TaskContract) -> TaskContract:
    obligations = list(primary.obligations)
    for obligation in fallback.obligations:
        if obligation.kind == "local_execution":
            execution_indexes = [
                index for index, existing in enumerate(obligations) if existing.kind == "local_execution"
            ]
            if execution_indexes:
                required = int(obligation.params.get("min_successes", 1) or 1)
                observed = sum(
                    int(obligations[index].params.get("min_successes", 1) or 1)
                    for index in execution_indexes
                )
                if observed < required:
                    index = execution_indexes[-1]
                    existing = obligations[index]
                    obligations[index] = ContractObligation(
                        id=existing.id,
                        kind=existing.kind,
                        description=existing.description,
                        evidence=existing.evidence or obligation.evidence,
                        required=existing.required,
                        params={
                            **obligation.params,
                            **existing.params,
                            "min_successes": int(existing.params.get("min_successes", 1) or 1)
                            + (required - observed),
                        },
                    )
                continue
        matching_index = next(
            (
                index
                for index, existing in enumerate(obligations)
                if existing.kind == obligation.kind
                and _normalized_path(existing.params.get("target_path"))
                == _normalized_path(obligation.params.get("target_path"))
            ),
            None,
        )
        if matching_index is None and obligation.kind in {
            "workspace_change",
            "workspace_delete",
            "workspace_discovery",
            "source_inspection",
            "source_report",
            "edit_review",
            "assistant_response",
            "visible_output",
            "test_evidence",
        }:
            same_kind = [
                index for index, existing in enumerate(obligations) if existing.kind == obligation.kind
            ]
            if len(same_kind) == 1:
                matching_index = same_kind[0]
        if matching_index is not None:
            existing = obligations[matching_index]
            merged_params = {**obligation.params, **existing.params}
            obligations[matching_index] = ContractObligation(
                id=existing.id,
                kind=existing.kind,
                description=existing.description,
                evidence=existing.evidence or obligation.evidence,
                required=existing.required,
                params=merged_params,
            )
            continue
        obligations.append(obligation)

    constraints = list(primary.constraints)
    existing_constraints = {(constraint.kind, constraint.first, constraint.second) for constraint in constraints}
    for constraint in fallback.constraints:
        key = (constraint.kind, constraint.first, constraint.second)
        if key in existing_constraints:
            continue
        constraints.append(constraint)
        existing_constraints.add(key)
    return TaskContract(
        intent=fallback.intent,
        obligations=tuple(obligations),
        constraints=tuple(_filter_constraints(constraints, obligations)),
    )


def _dedupe_obligations(obligations: list[ContractObligation]) -> list[ContractObligation]:
    seen: set[str] = set()
    unique: list[ContractObligation] = []
    for obligation in obligations:
        if obligation.id in seen:
            continue
        seen.add(obligation.id)
        unique.append(obligation)
    return unique


def _filter_constraints(
    constraints: list[ContractConstraint],
    obligations: list[ContractObligation],
) -> list[ContractConstraint]:
    by_id = {obligation.id: obligation for obligation in obligations}
    material_kinds = {
        "workspace_change",
        "workspace_delete",
        "workspace_discovery",
        "source_inspection",
        "source_report",
        "edit_review",
        "local_execution",
    }
    filtered: list[ContractConstraint] = []
    for constraint in constraints:
        first = by_id.get(constraint.first)
        second = by_id.get(constraint.second)
        if first is None or second is None:
            continue
        if first.kind not in material_kinds or second.kind not in material_kinds:
            continue
        filtered.append(constraint)
    return filtered


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


def _obligation_by_id(contract: TaskContract, obligation_id: str) -> ContractObligation | None:
    for obligation in contract.obligations:
        if obligation.id == obligation_id:
            return obligation
    return None


def _candidate_steps(obligation: ContractObligation, ledger: EvidenceLedger) -> list[int]:
    candidates = _candidate_observations(obligation, ledger)
    return [int(observation.get("step", 0)) for observation in candidates]


def _candidate_observations(obligation: ContractObligation, ledger: EvidenceLedger) -> tuple[dict[str, Any], ...]:
    target_path = _normalized_path(obligation.params.get("target_path"))
    if obligation.kind == "workspace_change":
        return tuple(_filter_path_observations(ledger.successful_changes, target_path))
    if obligation.kind == "workspace_delete":
        return tuple(_filter_path_observations(ledger.successful_deletes, target_path))
    if obligation.kind == "workspace_discovery":
        return ledger.successful_discovery
    if obligation.kind == "source_inspection":
        return tuple(_filter_path_observations(ledger.successful_reads, target_path))
    if obligation.kind == "source_report":
        reads = tuple(_filter_path_observations(ledger.successful_reads, target_path))
        if ledger.latest_change_step:
            return tuple(
                observation
                for observation in reads
                if int(observation.get("step", 0)) > ledger.latest_change_step
            )
        return reads
    if obligation.kind == "edit_review":
        return tuple(
            observation
            for observation in _filter_path_observations(ledger.successful_edit_reviews, target_path)
            if int(observation.get("step", 0)) > ledger.latest_change_step
        )
    if obligation.kind == "local_execution":
        command = str(obligation.params.get("command") or "").strip()
        observations = ledger.successful_runs
        if not target_path and not command:
            observations = _relevant_successful_runs(ledger)
        if target_path:
            command = ""
        return tuple(
            observation
            for observation in observations
            if _run_matches_target(observation, target_path) and _run_matches_command(observation, command)
        )
    if obligation.kind == "visible_output":
        return tuple(
            observation
            for observation in _relevant_successful_runs(ledger)
            if _has_meaningful_shell_output(observation)
        )
    if obligation.kind == "test_evidence":
        return tuple(
            observation
            for observation in _relevant_successful_runs(ledger)
            if _has_passing_test_evidence(observation)
        )
    return ()


def _filter_path_observations(observations: tuple[dict[str, Any], ...], target_path: str | None):
    for observation in observations:
        if target_path and _normalized_path(observation.get("result", {}).get("path")) != target_path:
            continue
        yield observation


def _run_matches_target(observation: dict[str, Any], target_path: str | None) -> bool:
    if not target_path:
        return True
    command = str(observation.get("action", {}).get("command", ""))
    return target_path in _normalized_command_paths(command)


def _run_matches_command(observation: dict[str, Any], command: str) -> bool:
    if not command:
        return True
    observed = " ".join(str(observation.get("action", {}).get("command", "")).split())
    expected = " ".join(command.split())
    return observed == expected


def command_target_paths(command: str) -> set[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths = {_normalized_path(token) for token in tokens if _normalized_path(token)}
    try:
        start_dir = tokens[tokens.index("-s") + 1]
        pattern = tokens[tokens.index("-p") + 1]
    except (ValueError, IndexError):
        return paths
    discovered_path = _normalized_path(f"{start_dir.rstrip('/')}/{pattern.lstrip('/')}")
    if discovered_path:
        paths.add(discovered_path)
    return paths


def _normalized_command_paths(command: str) -> set[str]:
    return command_target_paths(command)


def _target_path_from_command(command: str) -> str | None:
    for path in _normalized_command_paths(command):
        if path and path.endswith(".py") and not path.startswith("/") and ".." not in path.split("/"):
            return path
    return None


def _allowed_obligation_kinds() -> set[str]:
    return {
        "workspace_change",
        "workspace_delete",
        "workspace_discovery",
        "source_inspection",
        "source_report",
        "edit_review",
        "local_execution",
        "test_evidence",
        "visible_output",
        "assistant_response",
    }


def _sanitize_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    params: dict[str, Any] = {}
    for key in {"target_path", "command", "expected_text", "edit_scope"}:
        if key not in value:
            continue
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            params[key] = _truncate(_one_line(item), 500)
        elif item is None:
            params[key] = None
    if params.get("edit_scope") not in {None, "tests_only", "implementation", "whole_file"}:
        params.pop("edit_scope", None)
    min_successes = value.get("min_successes")
    if isinstance(min_successes, int):
        params["min_successes"] = max(1, min(min_successes, 100))
    elif isinstance(min_successes, str) and min_successes.isdigit():
        params["min_successes"] = max(1, min(int(min_successes), 100))
    if not params.get("target_path") and params.get("command"):
        target_path = _target_path_from_command(str(params["command"]))
        if target_path:
            params["target_path"] = target_path
    return params


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return normalized[:64]


def _normalized_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.replace("\\", "/").lstrip("./")


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
