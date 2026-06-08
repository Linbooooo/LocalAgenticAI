from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import AgentConfig
from .context import prepare_messages
from .ollama_client import OllamaClient, OllamaConnectionError
from .prompts import (
    ACTION_PROMPT,
    CONTRACT_PROMPT,
    EDIT_REPAIR_PROMPT,
    EDIT_REVIEW_PROMPT,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
    WORKSPACE_CHANGE_PROMPT,
)
from .skills import format_coding_skills, select_coding_skills
from .task_contract import (
    TaskContract,
    command_target_paths,
    contract_from_model_json,
    contract_missing as task_contract_missing,
    derive_task_contract,
    format_contract_evidence_for_final,
    format_contract_missing,
    format_task_contract,
)
from .tool_policy import extract_direct_shell_command
from .tools import WorkspaceTools


@dataclass
class AgentResult:
    content: str
    turns: int


@dataclass
class RouteDecision:
    mode: str
    requires_run: bool = False
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class CompletionCriteria:
    requires_run: bool = False
    requires_tests: bool = False
    requires_output: bool = False


@dataclass(frozen=True)
class TaskSpec:
    target_path: str | None = None
    language: str | None = None
    operation: str | None = None


class LocalAgent:
    ACTION_REPAIR_ATTEMPTS = 2

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
        self.tools = WorkspaceTools(config)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(workspace=str(config.workspace))}
        ]
        self.last_written_file: str | None = None
        self.last_shell_command: str | None = None
        self.last_shell_result: dict[str, Any] | None = None
        self._active_edit_baselines: dict[str, str] = {}

    def run(self, task: str) -> AgentResult:
        self.messages.append({"role": "user", "content": task})

        read_path = self._followup_read_file(task)
        if read_path:
            return self._read_file(read_path)

        command = extract_direct_shell_command(task)
        if command:
            return self._run_shell(command)

        route = self._route_task(task)
        if route.mode == "hardware":
            context = _format_json("Local hardware profile", self.tools.hardware_profile())
            return self._ask(context)
        if route.mode == "read":
            return self._act(task, route.mode, route.requires_run)
        if route.mode == "edit":
            return self._act(task, route.mode, route.requires_run)
        if route.mode == "shell":
            return self._act(task, route.mode, route.requires_run)
        return self._ask()

    def _ask(self, context: str | None = None) -> AgentResult:
        response = self._chat(context)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        self.messages.append({"role": "assistant", "content": content})
        return AgentResult(content=content, turns=1)

    def _route_task(self, task: str) -> RouteDecision:
        context = f"{ROUTER_PROMPT}\n\nUser request:\n{task}"
        try:
            response = self._protocol_chat(context, json_mode=True)
        except OllamaConnectionError:
            raise
        content = str(response.get("message", {}).get("content", "")).strip()
        route = _extract_json_object(content)
        if not isinstance(route, dict):
            return RouteDecision(mode="chat", confidence=0.0, reason="Router returned invalid JSON.")
        return _validate_route(route)

    def _act(self, task: str, intent: str, requires_run: bool = False) -> AgentResult:
        observations: list[dict[str, Any]] = []
        self._active_edit_baselines = {}
        criteria = _completion_criteria(task, requires_run, intent)
        task_spec = _task_spec(task, intent)
        task_spec = _resolve_edit_task_spec(task, intent, task_spec, self.last_written_file, self.config.workspace)
        if intent == "read":
            criteria = CompletionCriteria()
        contract = derive_task_contract(
            task=task,
            intent=intent,
            requires_run=criteria.requires_run,
            requires_tests=criteria.requires_tests,
            requires_output=criteria.requires_output,
            target_path=task_spec.target_path,
            operation=task_spec.operation,
        )
        contract = self._contract_task(task, intent, criteria, task_spec, contract)

        for step in range(1, self.config.max_steps + 1):
            action = self._contract_forced_action(contract, observations)
            if action is None:
                action = self._forced_action(intent, criteria, observations)
            if action is None:
                action = self._repair_recovery_action(criteria, observations, contract)
            if action is None:
                action = self._next_action(task, intent, criteria, observations, task_spec, contract)

            kind = str(action.get("action", "")).strip()
            if kind in {"answer", "finish"}:
                forced = self._forced_action(intent, criteria, observations)
                if forced is not None:
                    action = forced
                    kind = str(action.get("action", "")).strip()
                elif _should_reject_completion_action(kind, criteria, observations, contract):
                    recovery = self._repair_recovery_action(criteria, observations, contract)
                    if recovery is not None:
                        action = recovery
                        kind = str(action.get("action", "")).strip()
                    else:
                        observations.append(_completion_failure_observation(step, action, criteria, observations, contract))
                        if _should_stop_after_repair_stall(criteria, observations):
                            reply = _format_repair_stall_reply(observations)
                            self.messages.append({"role": "assistant", "content": reply})
                            return AgentResult(content=reply, turns=step)
                        continue
                else:
                    reply = str(action.get("message", "")).strip() or _format_observations(observations)
                    reply = _format_final_reply(reply, observations, contract)
                    self.messages.append({"role": "assistant", "content": reply})
                    return AgentResult(content=reply, turns=step)

            if _is_repeated_write(action, observations):
                forced = self._forced_action(intent, criteria, observations)
                if forced is not None:
                    action = forced
                else:
                    recovery = self._repair_recovery_action(criteria, observations, contract)
                    if recovery is not None:
                        action = recovery
                    else:
                        result = {
                            "ok": False,
                            "error": (
                                "Repeated identical write action. Rewriting the same content cannot repair "
                                "the failed verification; change the file, inspect context, or run a different command."
                            ),
                        }
                        observations.append(
                            {
                                "step": step,
                                "action": _public_action(action),
                                "signature": _action_signature(action),
                                "result": result,
                            }
                        )
                        if _should_stop_after_repair_stall(criteria, observations):
                            reply = _format_repair_stall_reply(observations)
                            self.messages.append({"role": "assistant", "content": reply})
                            return AgentResult(content=reply, turns=step)
                        continue

            if _is_repeated_unproductive_run(action, criteria, observations):
                recovery = self._repair_recovery_action(criteria, observations, contract)
                if recovery is not None:
                    action = recovery
                else:
                    result = {
                        "ok": False,
                        "blocked": True,
                        "error": (
                            "Repeated run_shell command cannot satisfy the requested output. "
                            "Edit the file to print results, create/run tests, or run a different command that displays results."
                        ),
                    }
                    observations.append(
                        {
                            "step": step,
                            "action": _public_action(action),
                            "signature": _action_signature(action),
                            "result": result,
                        }
                    )
                    if _should_stop_after_repair_stall(criteria, observations):
                        reply = _format_repair_stall_reply(observations)
                        self.messages.append({"role": "assistant", "content": reply})
                        return AgentResult(content=reply, turns=step)
                    continue

            if _is_contract_repeated_unsatisfied_run(action, contract, observations):
                error = (
                    "This command already ran after the latest edit without producing expected contract output. "
                    "Read or edit the target file before running the same command again."
                )
                result = {
                    "ok": False,
                    "error": error,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": error,
                }
                observations.append(
                    {
                        "step": step,
                        "action": _public_action(action),
                        "signature": _action_signature(action),
                        "result": result,
                    }
                )
                continue

            if _is_repeated_failed_run(action, observations):
                result = {
                    "ok": False,
                    "blocked": True,
                    "error": (
                        "Repeated run_shell command already failed after the latest edit. "
                        "Inspect the failure or change the file before running the same command again."
                    ),
                }
                observations.append(
                    {
                        "step": step,
                        "action": _public_action(action),
                        "signature": _action_signature(action),
                        "result": result,
                    }
                )
                if _should_stop_after_repair_stall(criteria, observations):
                    reply = _format_repair_stall_reply(observations)
                    self.messages.append({"role": "assistant", "content": reply})
                    return AgentResult(content=reply, turns=step)
                continue

            kind = str(action.get("action", "")).strip()
            if kind in {"answer", "finish"}:
                if _should_reject_completion_action(kind, criteria, observations, contract):
                    recovery = self._repair_recovery_action(criteria, observations, contract)
                    if recovery is not None:
                        action = recovery
                    else:
                        observations.append(_completion_failure_observation(step, action, criteria, observations, contract))
                        if _should_stop_after_repair_stall(criteria, observations):
                            reply = _format_repair_stall_reply(observations)
                            self.messages.append({"role": "assistant", "content": reply})
                            return AgentResult(content=reply, turns=step)
                        continue
                reply = str(action.get("message", "")).strip() or _format_observations(observations)
                reply = _format_final_reply(reply, observations, contract)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)

            try:
                if kind == "edit_review":
                    result = self._review_edit(task, action, observations)
                elif kind == "edit_repair":
                    result = self._repair_edit(task, action, observations)
                else:
                    result = self._apply_action(action, _tool_intent_for_action(intent, action, contract))
            except (TypeError, ValueError, OSError) as exc:
                result = {"ok": False, "error": str(exc)}
            observations.append(
                {
                    "step": step,
                    "action": _public_action(action),
                    "signature": _action_signature(action),
                    "result": result,
                }
            )
            if result.get("ok") and result.get("path"):
                self.last_written_file = str(result["path"])
            if action.get("action") == "run_shell":
                self.last_shell_command = str(action.get("command", "")).strip()
                self.last_shell_result = result
            if _should_finish_with_assistant_response(criteria, observations, contract):
                reply = self._assistant_response_reply(task, observations, contract)
                reply = _format_final_reply(reply, observations, contract)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)
            if _should_stop_after_simple_edit(task, intent, criteria, observations, contract):
                reply = _format_final_reply(_simple_edit_reply(observations[-1]), observations, contract)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)
            if _should_stop_after_success(intent, criteria, observations, contract):
                reply = _format_final_reply("Completed and verified successfully.", observations, contract)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)
            if _should_stop_after_contract_success(intent, criteria, observations, contract):
                reply = _format_final_reply("Completed all requested local steps.", observations, contract)
                self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)
            if not result.get("ok") and _should_stop_after_failure(result):
                reply = _format_observations(observations)
                if _is_user_declined_result(result):
                    _drop_latest_user_message(self.messages, task)
                else:
                    self.messages.append({"role": "assistant", "content": reply})
                return AgentResult(content=reply, turns=step)

        reply = f"Stopped after {self.config.max_steps} action steps.\n\n{_format_observations(observations)}"
        self.messages.append({"role": "assistant", "content": reply})
        return AgentResult(content=reply, turns=self.config.max_steps)

    def _next_action(
        self,
        task: str,
        intent: str,
        criteria: CompletionCriteria,
        observations: list[dict[str, Any]],
        task_spec: TaskSpec,
        contract: TaskContract,
    ) -> dict[str, Any]:
        pending_change = _pending_workspace_change(contract, observations)
        if pending_change is not None and _has_read_contract_target(pending_change, observations):
            constrained = self._constrained_workspace_change_action(task, pending_change, observations)
            if constrained is not None:
                return constrained

        protocol_error: str | None = None
        last_content = ""

        for _ in range(self.ACTION_REPAIR_ATTEMPTS + 1):
            response = self._protocol_chat(
                self._action_context(task, intent, criteria, observations, protocol_error, task_spec, contract),
                json_mode=True,
                json_temperature=0.0,
            )
            last_content = str(response.get("message", {}).get("content", "")).strip()
            action = _extract_json_object(last_content)
            if not isinstance(action, dict):
                salvaged = _salvage_action_from_text(task, intent, last_content, task_spec)
                if salvaged is not None:
                    return salvaged
                protocol_error = _protocol_error("No valid JSON action object was found.", last_content)
                continue

            validation_error = _validate_action(action, intent, criteria, observations, self.config.workspace)
            if validation_error is None:
                validation_error = _validate_action_against_contract(action, contract, observations)
            if validation_error is None:
                return action
            protocol_error = _protocol_error(validation_error, last_content)

        salvaged = _salvage_action_from_text(task, intent, last_content, task_spec)
        if salvaged is not None and _validate_action(salvaged, intent, criteria, observations, self.config.workspace) is None:
            return salvaged
        recovery = self._repair_recovery_action(criteria, observations, contract)
        if recovery is not None:
            return recovery
        if _repair_required(criteria, observations):
            return {
                "action": "answer",
                "message": "No valid corrective local action was produced yet after the failed verification.",
            }
        return {
            "action": "answer",
            "message": "I could not get a valid local action from the model after retrying the action protocol.",
        }

    def _constrained_workspace_change_action(
        self,
        task: str,
        obligation: Any,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_path = _contract_target_path(obligation)
        if not target_path:
            return None
        source = self.tools.read_file(target_path, max_lines=600)
        if not source.get("ok"):
            return None
        base_context = "\n\n".join(
            [
                WORKSPACE_CHANGE_PROMPT,
                f"User request:\n{task}",
                f"Target path: {target_path}",
                f"Outstanding change:\n{obligation.description}",
                _format_json("Change parameters", obligation.params),
                f"Current source:\n{source.get('raw_content', source.get('content', ''))}",
                _format_json("Previous action results", observations[-8:]),
            ]
        )
        protocol_error: str | None = None
        for _ in range(self.ACTION_REPAIR_ATTEMPTS + 1):
            context = base_context if protocol_error is None else f"{base_context}\n\n{protocol_error}"
            response = self._protocol_chat(context, json_mode=True, json_temperature=0.0)
            content = str(response.get("message", {}).get("content", "")).strip()
            action = _extract_json_object(content)
            if not isinstance(action, dict):
                protocol_error = _protocol_error("No valid JSON edit action was found.", content)
                continue
            error = _validate_action(action, "edit", CompletionCriteria(), observations, self.config.workspace)
            if error is None:
                error = _validate_action_against_contract(
                    action,
                    TaskContract(intent="edit", obligations=(obligation,)),
                    observations,
                )
            if error is None and action.get("action") in {"write_file", "replace_in_file"}:
                return action
            protocol_error = _protocol_error(
                error or "Only write_file or replace_in_file is allowed in this phase.",
                content,
            )
        return None

    def _repair_recovery_action(
        self,
        criteria: CompletionCriteria,
        observations: list[dict[str, Any]],
        contract: TaskContract | None = None,
    ) -> dict[str, Any] | None:
        if not _repair_required(criteria, observations):
            return None
        if not self.last_written_file:
            return None
        failed_step = _last_failed_run_step_after_latest_change(observations)
        run_action = self._alternate_run_action_after_failure(failed_step, observations)
        if run_action is not None:
            return run_action
        if _last_action_step(observations, "read_file") > failed_step:
            action = self._deterministic_test_repair_action(observations, contract)
            if action is not None:
                return action
            return None
        return {
            "action": "read_file",
            "path": self.last_written_file,
            "start_line": 1,
            "max_lines": 240,
        }

    def _deterministic_test_repair_action(
        self,
        observations: list[dict[str, Any]],
        contract: TaskContract | None,
    ) -> dict[str, Any] | None:
        if _contract_edit_scope(contract) != "tests_only":
            return None
        if not self.last_written_file or not self.last_written_file.endswith(".py"):
            return None
        failed = _last_failed_run_after_latest_change(observations)
        if "AssertionError" not in _shell_output_text(failed):
            return None
        path = self.config.workspace / self.last_written_file
        repaired = _repair_python_assert_expected_literals(path)
        if repaired is None:
            return None
        return {"action": "write_file", "path": self.last_written_file, "content": repaired}

    def _alternate_run_action_after_failure(
        self,
        failed_step: int,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.last_written_file or not self.last_written_file.endswith(".py"):
            return None
        file_path = self.config.workspace / self.last_written_file
        if not _python_file_has_visible_output(file_path) or _python_file_requires_stdin(file_path):
            return None
        command = _python_run_command_for_path(self.last_written_file, file_path)
        failed_command = str(_last_failed_run_after_latest_change(observations).get("action", {}).get("command", "")).strip()
        if _command_signature(failed_command) == _command_signature(command):
            return None
        if _has_run_command_after_step(command, failed_step, observations):
            return None
        return {"action": "run_shell", "command": command, "timeout_seconds": 120}

    def _forced_action(
        self,
        intent: str,
        criteria: CompletionCriteria,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if intent != "edit" or not criteria.requires_run:
            return None
        if not self.last_written_file or not self.last_written_file.endswith(".py"):
            return None
        if _last_successful_change_step(observations) <= _last_action_step(observations, "run_shell"):
            return None
        file_path = self.config.workspace / self.last_written_file
        if not file_path.exists():
            return None
        if criteria.requires_tests and not _is_python_test_file(self.last_written_file) and not _python_file_has_inline_tests(file_path):
            return None
        if criteria.requires_output and not criteria.requires_tests and not _python_file_has_visible_output(file_path):
            return None
        if _python_file_requires_stdin(file_path):
            return None
        return {
            "action": "run_shell",
            "command": _python_run_command_for_path(self.last_written_file, file_path),
            "timeout_seconds": 120,
        }

    def _contract_forced_action(
        self,
        contract: TaskContract,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if _last_failed_run_step_after_latest_change(observations) > _last_successful_run_step_after_latest_change(observations):
            return None
        missing = task_contract_missing(contract, observations)
        if not missing:
            return None

        review_obligation = next(
            (obligation for obligation in contract.obligations if obligation.kind == "edit_review" and obligation.required),
            None,
        )
        if review_obligation is not None:
            edit_scope = _contract_edit_scope(contract)
            latest_change = _last_successful_change_step(observations)
            if latest_change == 0:
                inspection = next(
                    (
                        obligation
                        for obligation in contract.obligations
                        if obligation.kind == "source_inspection" and obligation.required
                    ),
                    None,
                )
                if inspection is not None and not _contract_observation_steps(inspection, observations):
                    action = self._read_action_for_contract_obligation(inspection)
                    if action is not None:
                        return action
            elif _last_action_step(observations, "edit_review") <= latest_change:
                action = self._edit_review_action_for_contract_obligation(review_obligation, edit_scope)
                if action is not None:
                    return action
            else:
                action = self._edit_repair_action_after_failed_review(review_obligation, observations, edit_scope)
                if action is not None:
                    return action

        pending_change = _pending_workspace_change(contract, observations)
        if pending_change is not None and not _has_read_contract_target(pending_change, observations):
            target_path = _contract_target_path(pending_change)
            if target_path:
                file_path = self.config.workspace / target_path
                try:
                    if file_path.resolve().is_relative_to(self.config.workspace) and file_path.is_file():
                        return {"action": "read_file", "path": target_path, "start_line": 1, "max_lines": 200}
                except ValueError:
                    pass

        for constraint in contract.constraints:
            if constraint.kind != "before":
                continue
            first = _contract_obligation_by_id(contract, constraint.first)
            second = _contract_obligation_by_id(contract, constraint.second)
            if first is None or second is None or second.kind != "local_execution":
                continue
            first_steps = _contract_observation_steps(first, observations)
            second_steps = _contract_observation_steps(second, observations)
            if not first_steps:
                action = self._read_action_for_contract_obligation(first)
                if action is not None:
                    return action
            if first_steps and not any(first_step < second_step for first_step in first_steps for second_step in second_steps):
                if _has_unsatisfied_recent_run_for_obligation(second, observations):
                    continue
                action = self._run_action_for_contract_obligation(second)
                if action is not None:
                    return action

        for obligation in contract.obligations:
            if obligation.kind == "source_report" and obligation.required:
                matching_steps = _contract_observation_steps(obligation, observations)
                if matching_steps:
                    continue
                action = self._read_action_for_contract_obligation(obligation)
                if action is not None:
                    return action

        for obligation in contract.obligations:
            if obligation.kind != "local_execution" or not obligation.required:
                continue
            if _has_missing_workspace_change(contract, observations) and not _must_run_before_workspace_change(
                contract, obligation
            ):
                continue
            min_successes = int(obligation.params.get("min_successes", 1) or 1)
            matching_steps = _contract_observation_steps(obligation, observations)
            if len(matching_steps) >= min_successes:
                continue
            if _has_unsatisfied_recent_run_for_obligation(obligation, observations):
                continue
            action = self._run_action_for_contract_obligation(obligation)
            if action is not None:
                return action
        return None

    def _edit_review_action_for_contract_obligation(
        self,
        obligation: Any,
        edit_scope: str | None = None,
    ) -> dict[str, Any] | None:
        target_path = _contract_target_path(obligation) or self.last_written_file
        if not target_path:
            return None
        file_path = self.config.workspace / target_path
        try:
            if not file_path.resolve().is_relative_to(self.config.workspace):
                return None
        except ValueError:
            return None
        if not file_path.is_file():
            return None
        return {"action": "edit_review", "path": target_path, "edit_scope": edit_scope}

    def _edit_repair_action_after_failed_review(
        self,
        obligation: Any,
        observations: list[dict[str, Any]],
        edit_scope: str | None = None,
    ) -> dict[str, Any] | None:
        failed_review = _latest_failed_edit_review_after_latest_change(observations)
        failed_step = int(failed_review.get("step", 0))
        if failed_step <= 0:
            return None
        latest_change = _last_successful_change_step(observations)
        if latest_change > failed_step:
            return None
        target_path = _contract_target_path(obligation) or self.last_written_file
        if not target_path:
            return None
        if _last_action_step(observations, "read_file") <= failed_step:
            read_action = self._read_action_for_contract_obligation(obligation)
            if read_action is None:
                read_action = {"action": "read_file", "path": target_path, "start_line": 1, "max_lines": 240}
            return read_action
        if _last_action_step(observations, "edit_repair") > failed_step:
            return None
        return {"action": "edit_repair", "path": target_path, "edit_scope": edit_scope}

    def _run_action_for_contract_obligation(self, obligation: Any) -> dict[str, Any] | None:
        target_path = str(obligation.params.get("target_path") or "").strip()
        if not target_path or not target_path.endswith(".py"):
            return None
        file_path = self.config.workspace / target_path
        try:
            if not file_path.resolve().is_relative_to(self.config.workspace):
                return None
        except ValueError:
            return None
        if not file_path.exists():
            return None
        if _python_file_requires_stdin(file_path):
            return None
        return {
            "action": "run_shell",
            "command": _python_run_command_for_path(target_path, file_path),
            "timeout_seconds": 120,
        }

    def _read_action_for_contract_obligation(self, obligation: Any) -> dict[str, Any] | None:
        if obligation.kind not in {"source_inspection", "source_report"}:
            return None
        target_path = _contract_target_path(obligation) or self.last_written_file
        if not target_path:
            return None
        file_path = self.config.workspace / target_path
        try:
            if not file_path.resolve().is_relative_to(self.config.workspace):
                return None
        except ValueError:
            return None
        if not file_path.exists():
            return None
        return {"action": "read_file", "path": target_path, "start_line": 1, "max_lines": 200}

    def _review_edit(
        self,
        task: str,
        action: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = str(action.get("path", "")).strip()
        if not path:
            return {"ok": False, "error": "Edit review requires a target path."}
        file_path = (self.config.workspace / path).resolve()
        if not file_path.is_relative_to(self.config.workspace):
            return {"ok": False, "error": f"Path escapes workspace: {path}"}
        if not file_path.is_file():
            return {"ok": False, "error": f"Path does not exist: {path}"}

        edited_source = file_path.read_text(encoding="utf-8", errors="replace")
        original_source = self._active_edit_baselines.get(path) or _source_before_latest_change(path, observations)
        edit_scope = str(action.get("edit_scope") or "").strip() or None
        scope_issues = _deterministic_edit_scope_issues(path, edit_scope, original_source, edited_source)
        if scope_issues:
            return {
                "ok": False,
                "path": path,
                "satisfied": False,
                "reason": "The edit changed code outside the requested scope.",
                "missing": scope_issues,
                "error": f"Edit review found unmet requirements: {'; '.join(scope_issues)}",
            }
        source_diff = "\n".join(
            difflib.unified_diff(
                original_source.splitlines(),
                edited_source.splitlines(),
                fromfile=f"{path} (before)",
                tofile=f"{path} (after)",
                lineterm="",
            )
        )
        review_context = "\n\n".join(
            [
                EDIT_REVIEW_PROMPT,
                f"User request:\n{task}",
                f"Target path: {path}",
                f"Requested edit scope: {edit_scope or 'unspecified'}",
                f"Source diff:\n{_truncate_middle(source_diff or '[no textual diff]', 10000)}",
                f"Original source before this edit:\n{_truncate_middle(original_source or '[not available]', 6000)}",
                f"Edited source:\n{_truncate_middle(edited_source, 6000)}",
                _format_edit_review_history(observations),
            ]
        )
        protocol_error: str | None = None
        for _ in range(self.ACTION_REPAIR_ATTEMPTS + 1):
            context = review_context if protocol_error is None else f"{review_context}\n\n{protocol_error}"
            response = self._protocol_chat(context, json_mode=True, json_temperature=0.0)
            content = str(response.get("message", {}).get("content", "")).strip()
            review = _extract_json_object(content)
            if not isinstance(review, dict) or not isinstance(review.get("satisfied"), bool):
                protocol_error = _protocol_error(
                    "Edit review requires a JSON boolean field named satisfied.",
                    content,
                )
                continue

            reason = _truncate(_one_line(str(review.get("reason", ""))), 600)
            missing = [
                _truncate(_one_line(str(item)), 300)
                for item in review.get("missing", [])
                if isinstance(item, str) and _one_line(item)
            ][:12]
            if not review["satisfied"] and _review_failure_is_external_evidence_only(reason, missing):
                return {
                    "ok": True,
                    "path": path,
                    "satisfied": True,
                    "reason": "Source review passed; runtime evidence is validated by the execution ledger.",
                    "missing": [],
                }
            if review["satisfied"]:
                return {
                    "ok": True,
                    "path": path,
                    "satisfied": True,
                    "reason": reason or "The edited source satisfies the requested transformation.",
                    "missing": [],
                }
            detail = "; ".join(missing) or reason or "The requested transformation is incomplete."
            return {
                "ok": False,
                "path": path,
                "satisfied": False,
                "reason": reason,
                "missing": missing,
                "error": f"Edit review found unmet requirements: {detail}",
            }
        return {"ok": False, "path": path, "error": "The model did not return a valid edit review."}

    def _repair_edit(
        self,
        task: str,
        action: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = str(action.get("path", "")).strip()
        if not path:
            return {"ok": False, "error": "Edit repair requires a target path."}
        file_path = (self.config.workspace / path).resolve()
        if not file_path.is_relative_to(self.config.workspace):
            return {"ok": False, "error": f"Path escapes workspace: {path}"}
        if not file_path.is_file():
            return {"ok": False, "error": f"Path does not exist: {path}"}

        current_source = file_path.read_text(encoding="utf-8", errors="replace")
        original_source = self._active_edit_baselines.get(path) or _source_before_latest_change(path, observations)
        edit_scope = str(action.get("edit_scope") or "").strip() or None
        repair_context = "\n\n".join(
            [
                EDIT_REPAIR_PROMPT,
                f"User request:\n{task}",
                f"Target path: {path}",
                f"Requested edit scope: {edit_scope or 'unspecified'}",
                f"Failed review:\n{_format_latest_failed_edit_review(observations)}",
                f"Original source before this edit:\n{_truncate_middle(original_source or '[not available]', 6000)}",
                f"Current source:\n{_truncate_middle(current_source, 9000)}",
                _format_edit_review_history(observations),
            ]
        )
        protocol_error: str | None = None
        for _ in range(self.ACTION_REPAIR_ATTEMPTS + 1):
            context = repair_context if protocol_error is None else f"{repair_context}\n\n{protocol_error}"
            response = self._protocol_chat(context, json_mode=True, json_temperature=0.0)
            content = str(response.get("message", {}).get("content", "")).strip()
            repair = _extract_json_object(content)
            if not isinstance(repair, dict):
                protocol_error = _protocol_error("Edit repair requires a JSON object.", content)
                continue
            repair_path = str(repair.get("path", path)).strip() or path
            if repair_path.replace("\\", "/").lstrip("./") != path.replace("\\", "/").lstrip("./"):
                protocol_error = _protocol_error("Edit repair must target the requested path.", content)
                continue
            if "content" not in repair:
                protocol_error = _protocol_error("Edit repair requires complete file content.", content)
                continue
            self._remember_edit_baseline(path)
            return self.tools.write_file(path, _normalize_written_content(str(repair["content"])))
        return {"ok": False, "path": path, "error": "The model did not return a valid edit repair."}

    def _apply_action(self, action: dict[str, Any], intent: str) -> dict[str, Any]:
        kind = action.get("action")
        if kind == "list_files":
            return self.tools.list_files(
                str(action.get("path", ".")),
                int(action.get("max_depth", 4)),
                int(action.get("limit", 200)),
            )
        if kind == "read_file":
            return self.tools.read_file(
                str(action.get("path", "")),
                int(action.get("start_line", 1)),
                int(action.get("max_lines", 200)),
            )
        if kind == "search_text":
            return self.tools.search_text(
                str(action.get("pattern", "")),
                str(action.get("path", ".")),
                str(action.get("file_glob", "*")),
                _bool_value(action.get("case_sensitive", False)),
            )
        if kind == "write_file":
            if intent != "edit":
                return {"ok": False, "error": "File edits are only allowed for edit requests."}
            path = str(action.get("path", "")).strip()
            if not path:
                return {"ok": False, "error": "Missing file path."}
            self._remember_edit_baseline(path)
            return self.tools.write_file(path, _normalize_written_content(str(action.get("content", ""))))
        if kind == "replace_in_file":
            if intent != "edit":
                return {"ok": False, "error": "File edits are only allowed for edit requests."}
            old = str(action.get("old", ""))
            if old == "":
                return {"ok": False, "error": "Replacement target is empty."}
            self._remember_edit_baseline(str(action.get("path", "")))
            return self.tools.replace_in_file(
                str(action.get("path", "")),
                old,
                str(action.get("new", "")),
                int(action.get("max_replacements", 1)),
            )
        if kind == "delete_file":
            if intent != "edit":
                return {"ok": False, "error": "File deletion is only allowed for edit requests."}
            path = str(action.get("path", "")).strip()
            if not path:
                return {"ok": False, "error": "Missing file path."}
            return self.tools.delete_file(path)
        if kind == "run_shell":
            if intent not in {"edit", "shell"}:
                return {"ok": False, "error": "Shell commands are only allowed for shell or edit requests."}
            command = str(action.get("command", "")).strip()
            if not command:
                return {"ok": False, "error": "Missing shell command."}
            stdin = action.get("stdin")
            return self.tools.run_shell(
                command,
                int(action.get("timeout_seconds", 120)),
                None if stdin is None else str(stdin),
            )
        return {"ok": False, "error": f"Unknown action: {kind}"}

    def _remember_edit_baseline(self, path: str) -> None:
        normalized = path.replace("\\", "/").lstrip("./")
        if not normalized or normalized in self._active_edit_baselines:
            return
        file_path = (self.config.workspace / normalized).resolve()
        if not file_path.is_relative_to(self.config.workspace) or not file_path.is_file():
            return
        self._active_edit_baselines[normalized] = file_path.read_text(encoding="utf-8", errors="replace")

    def _run_shell(self, command: str) -> AgentResult:
        result = self.tools.run_shell(command)
        self.last_shell_command = command
        self.last_shell_result = result
        self.messages.append({"role": "assistant", "content": _format_shell_result(result)})
        return AgentResult(content=_format_shell_result(result), turns=1)

    def _read_file(self, path: str) -> AgentResult:
        result = self.tools.read_file(path, max_lines=400)
        content = _format_read_result(result)
        self.messages.append({"role": "assistant", "content": content})
        return AgentResult(content=content, turns=1)

    def _followup_read_file(self, task: str) -> str | None:
        if not _looks_like_read_file_request(task):
            return None
        if _looks_like_edit_or_test_creation_request(task.lower()):
            return None

        listing = self.tools.list_files(max_depth=4, limit=300)
        files = [str(path) for path in listing.get("files", [])] if listing.get("ok") else []
        matches = _matching_code_files(task, files)
        if matches:
            return matches[0]
        if self.last_written_file:
            return self.last_written_file
        return None

    def _workspace_context(self, task: str) -> str:
        context, _ = self._workspace_context_and_files(task)
        return context

    def _workspace_context_and_files(self, task: str) -> tuple[str, list[str]]:
        listing = self.tools.list_files(max_depth=3, limit=120)
        files = listing.get("files", []) if listing.get("ok") else []
        selected = _select_context_files(task, files)
        parts = [_format_json("Workspace files", listing)]
        for path in selected:
            result = self.tools.read_file(path, max_lines=160)
            if result.get("ok"):
                parts.append(f"File: {path}\n{result.get('raw_content', result['content'])}")
        return "\n\n".join(parts), [str(path) for path in files]

    def _action_context(
        self,
        task: str,
        intent: str,
        criteria: CompletionCriteria,
        observations: list[dict[str, Any]],
        protocol_error: str | None = None,
        task_spec: TaskSpec | None = None,
        contract: TaskContract | None = None,
    ) -> str:
        workspace_context, workspace_files = self._workspace_context_and_files(task)
        skills = format_coding_skills(select_coding_skills(task, intent, workspace_files, observations))
        parts = [
            ACTION_PROMPT,
            f"User request:\n{task}",
            f"Request intent: {intent}",
            _format_agent_state(self),
            _format_completion_criteria(criteria),
            _format_task_spec(task_spec),
            format_task_contract(contract) if contract is not None else "",
            _format_next_action_constraint(contract, observations) if contract is not None else "",
            skills,
            workspace_context,
        ]
        if intent == "read":
            parts.append("For this read request, do not edit files or run shell commands.")
        elif intent == "shell":
            parts.append("For this shell request, do not edit files unless the user explicitly asks for edits.")
        if protocol_error:
            parts.append(protocol_error)
        if observations:
            parts.append(_format_json("Previous action results", observations))
        repair_guidance = _format_repair_guidance(criteria, observations)
        if repair_guidance:
            parts.append(repair_guidance)
        missing = _completion_missing(criteria, observations)
        if missing and not repair_guidance:
            parts.append(_format_completion_missing(missing))
        if contract is not None and not repair_guidance:
            contract_missing = task_contract_missing(contract, observations)
            if contract_missing:
                parts.append(format_contract_missing(contract_missing))
        return "\n\n".join(parts)

    def _contract_task(
        self,
        task: str,
        intent: str,
        criteria: CompletionCriteria,
        task_spec: TaskSpec,
        fallback: TaskContract,
    ) -> TaskContract:
        if self.config.contract_mode == "fallback":
            return fallback

        listing = self.tools.list_files(max_depth=3, limit=120)
        parts = [
            CONTRACT_PROMPT,
            f"User request:\n{task}",
            f"Request intent: {intent}",
            _format_agent_state(self),
            _format_completion_criteria(criteria),
            _format_task_spec(task_spec),
            _format_json("Workspace files", listing),
            _format_json(
                "Fallback contract",
                {
                    "obligations": [
                        {
                            "id": obligation.id,
                            "kind": obligation.kind,
                            "description": obligation.description,
                            "required": obligation.required,
                            "params": obligation.params,
                            "evidence": list(obligation.evidence),
                        }
                        for obligation in fallback.obligations
                    ],
                    "constraints": [
                        {
                            "kind": constraint.kind,
                            "first": constraint.first,
                            "second": constraint.second,
                            "description": constraint.description,
                        }
                        for constraint in fallback.constraints
                    ],
                },
            ),
        ]
        try:
            response = self._protocol_chat("\n\n".join(parts), json_mode=True, json_temperature=0.0)
        except OllamaConnectionError:
            raise
        content = str(response.get("message", {}).get("content", "")).strip()
        contract_json = _extract_json_object(content)
        return contract_from_model_json(contract_json, fallback=fallback, task=task)

    def _assistant_response_reply(
        self,
        task: str,
        observations: list[dict[str, Any]],
        contract: TaskContract,
    ) -> str:
        context = "\n\n".join(
            [
                "All non-conversational local task obligations are satisfied.",
                f"User request:\n{task}",
                format_task_contract(contract),
                _format_json("Observed local actions", observations),
                (
                    "Write the final answer to the user. Include the requested conversational response, "
                    "and base any local-work claims only on the observed actions. If the user asked a "
                    "conversational question directed at you, answer it directly in first person; do not "
                    "ask the question back or add a follow-up question. Do not include command transcripts, "
                    "shell prompts, code blocks, or a Command results section because the harness will append "
                    "observed command results."
                ),
            ]
        )
        response = self._protocol_chat(context)
        content = str(response.get("message", {}).get("content", "")).strip()
        content = _strip_trailing_question(content)
        return content or "Completed the local work and addressed the requested response."

    def _chat(
        self,
        context: str | None = None,
        *,
        json_mode: bool = False,
        json_temperature: float | None = None,
    ) -> dict[str, Any]:
        messages = self._messages_with_context(context)
        return self._send_chat(messages, json_mode=json_mode, json_temperature=json_temperature)

    def _protocol_chat(
        self,
        context: str,
        *,
        json_mode: bool = False,
        json_temperature: float | None = None,
    ) -> dict[str, Any]:
        messages = [self.messages[0], {"role": "system", "content": context}]
        return self._send_chat(messages, json_mode=json_mode, json_temperature=json_temperature)

    def _send_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        json_temperature: float | None = None,
    ) -> dict[str, Any]:
        options = self.config.ollama_options()
        if json_mode:
            options = {**options, "temperature": 0.0 if json_temperature is None else json_temperature, "top_p": 1.0}
        while True:
            try:
                return self.client.chat(
                    model=self.config.model,
                    messages=prepare_messages(messages, self.config.context_budget_tokens()),
                    options=options,
                    keep_alive=self.config.keep_alive,
                    response_format="json" if json_mode else None,
                )
            except OllamaConnectionError as exc:
                message = str(exc).lower()
                if "memory" not in message and "out of" not in message:
                    raise
                if self.config.num_ctx <= self.config.min_num_ctx:
                    raise
                self.config.num_ctx = max(self.config.min_num_ctx, self.config.num_ctx // 2)

    def _messages_with_context(self, context: str | None) -> list[dict[str, Any]]:
        if not context:
            return self.messages
        return [self.messages[0], {"role": "system", "content": context}, *self.messages[1:]]


def _select_context_files(task: str, files: list[str]) -> list[str]:
    lower_task = task.lower()
    referenced = [
        path
        for path in files
        if Path(path).name.lower() in lower_task or path.lower() in lower_task
    ]
    defaults = {"pyproject.toml", "package.json", "cargo.toml"}
    if not referenced:
        defaults.add("readme.md")
    selected: list[str] = list(dict.fromkeys(referenced[:4]))
    for path in files:
        name = Path(path).name.lower()
        if name in defaults and path not in selected:
            selected.append(path)
        if len(selected) >= 6:
            break
    return selected


def _looks_like_read_file_request(task: str) -> bool:
    text = task.lower()
    if not re.search(r"\b(show|display|read|open|view)\b", text):
        return False
    return any(marker in text for marker in {"code", "file", "script", ".py"})


def _matching_code_files(task: str, files: list[str]) -> list[str]:
    normalized_task = _lookup_text(task)
    candidates: list[str] = []
    for path in files:
        file_path = Path(path)
        if file_path.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp", ".c"}:
            continue
        name = _lookup_text(file_path.name)
        stem = _lookup_text(file_path.stem)
        full_path = _lookup_text(path)
        if stem and (stem in normalized_task or name in normalized_task or full_path in normalized_task):
            candidates.append(path)
    return sorted(candidates, key=lambda path: (len(Path(path).parts), len(path), path))


def _lookup_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _looks_like_edit_or_test_creation_request(text: str) -> bool:
    return any(
        re.search(rf"\b{word}\b", text)
        for word in {
            "write",
            "create",
            "add",
            "modify",
            "update",
            "change",
            "replace",
            "fix",
            "debug",
            "repair",
            "implement",
            "solve",
            "solves",
            "generate",
            "test",
            "tests",
            "case",
            "cases",
            "unittest",
            "pytest",
            "run",
            "execute",
            "rerun",
            "delete",
            "remove",
        }
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip(), *_fenced_blocks(text)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                continue
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            return value
    return None


def _validate_route(route: dict[str, Any]) -> RouteDecision:
    mode = str(route.get("mode", "chat")).strip().lower()
    allowed = {"chat", "read", "edit", "shell", "hardware"}
    if mode not in allowed:
        return RouteDecision(mode="chat", confidence=0.0, reason=f"Unknown route mode: {mode}")

    try:
        confidence = float(route.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    reason = str(route.get("reason", "")).strip()
    requires_run = _bool_value(route.get("requires_run", False))

    if confidence < 0.70 and mode in {"edit", "shell"}:
        return RouteDecision(
            mode="chat",
            requires_run=False,
            confidence=confidence,
            reason=reason or "Low-confidence action route downgraded to chat.",
        )
    return RouteDecision(mode=mode, requires_run=requires_run, confidence=confidence, reason=reason)


def _completion_criteria(task: str, route_requires_run: bool, intent: str = "edit") -> CompletionCriteria:
    text = task.lower()
    requires_tests = bool(
        "tests/" in text
        or re.search(r"\b(test cases?|unit tests?|unittests?|pytest|unittest)\b", text)
        or re.search(r"\b(write|create|add|generate)\b.*\btests\b", text)
    )
    explicit_execution = _looks_like_explicit_execution_request(text)
    requires_output = bool(
        re.search(r"\b(display|show|print|report)\b.{0,80}\b(output|results?)\b", text)
        or re.search(r"\b(output|results?)\b.{0,80}\b(display|show|print|report)\b", text)
        or (explicit_execution and re.search(r"\b(display|show|output|results?|report)\b", text))
    )
    return CompletionCriteria(
        requires_run=(route_requires_run and (intent == "shell" or explicit_execution)) or requires_tests or requires_output,
        requires_tests=requires_tests,
        requires_output=requires_output,
    )


def _looks_like_explicit_execution_request(text: str) -> bool:
    return bool(
        re.search(r"\b(run|execute|rerun|test|tests|testing|verify|verification)\b", text)
        or re.search(r"\bcheck\b.{0,80}\b(code|file|program|script|tests?|behavior|output|results?)\b", text)
        or re.search(r"\b(display|show|print|report)\b.{0,80}\b(output|results?)\b", text)
    )


def _task_spec(task: str, intent: str) -> TaskSpec:
    if intent != "edit":
        return TaskSpec()

    operation = _task_operation(task)
    target_path = _explicit_filename_from_task(task)
    if target_path is None and operation == "create":
        target_path = _filename_from_task(task)
    language = "python" if target_path and target_path.endswith(".py") else None
    return TaskSpec(
        target_path=target_path,
        language=language,
        operation=operation,
    )


def _resolve_edit_task_spec(
    task: str,
    intent: str,
    task_spec: TaskSpec,
    last_written_file: str | None,
    workspace: Path,
) -> TaskSpec:
    if intent != "edit":
        return task_spec

    target_path = task_spec.target_path
    operation = task_spec.operation
    if target_path is None and operation in {"update", "repair"} and last_written_file:
        target_path = last_written_file

    if target_path:
        candidate = (workspace / target_path).resolve()
        try:
            existing_target = candidate.is_relative_to(workspace.resolve()) and candidate.is_file()
        except ValueError:
            existing_target = False
        if existing_target and operation == "create":
            operation = "update"

    language = task_spec.language
    if language is None and target_path and target_path.endswith(".py"):
        language = "python"
    return TaskSpec(target_path=target_path, language=language, operation=operation)


def _task_operation(task: str) -> str | None:
    text = task.lower()
    if re.search(r"\b(fix|debug|repair)\b", text):
        return "repair"
    if re.search(r"\b(refactor|rewrite|modify|update|change)\b", text):
        return "update"
    if re.search(r"\b(write|create|add|generate|implement|solve|build)\b", text):
        return "create"
    return None


def _format_completion_criteria(criteria: CompletionCriteria) -> str:
    requirements: list[str] = []
    if criteria.requires_run:
        requirements.append("a successful local command must run after the latest edit")
    if criteria.requires_tests:
        requirements.append("test evidence must be observed from a test command or test output")
    if criteria.requires_output:
        requirements.append("the command must produce visible output/results")
    if not requirements:
        return "Completion requirements: no extra run/test/output evidence was requested."
    lines = ["Completion requirements:"]
    lines.extend(f"- {requirement}" for requirement in requirements)
    return "\n".join(lines)


def _format_task_spec(task_spec: TaskSpec | None) -> str:
    if not task_spec or not any((task_spec.target_path, task_spec.language, task_spec.operation)):
        return "Planned artifact: none inferred from the request."

    lines = ["Planned artifact:"]
    if task_spec.target_path:
        lines.append(f"- target_path: {task_spec.target_path}")
    if task_spec.language:
        lines.append(f"- language: {task_spec.language}")
    if task_spec.operation:
        lines.append(f"- operation: {task_spec.operation}")
    if task_spec.target_path:
        lines.append("- Use target_path for write_file when creating the requested artifact unless workspace context proves a better path.")
    return "\n".join(lines)


def _completion_missing(criteria: CompletionCriteria, observations: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    if not criteria.requires_run:
        return missing

    runs = _successful_runs_after_latest_change(observations)
    if not runs:
        missing.append("No successful run_shell action has run after the latest edit.")
        return missing
    if criteria.requires_tests and not any(_has_passing_test_evidence(observation) for observation in runs):
        missing.append("Tests were requested, but no passing test command/output evidence was observed.")
    if criteria.requires_output and not any(_has_meaningful_shell_output(observation) for observation in runs):
        missing.append("Displayed results/output were requested, but the successful command produced no output.")
    return missing


def _format_completion_missing(missing: list[str]) -> str:
    lines = ["Unmet completion requirements:"]
    lines.extend(f"- {item}" for item in missing)
    lines.append("Choose another local action that produces the missing evidence before using finish.")
    return "\n".join(lines)


def _repair_required(criteria: CompletionCriteria, observations: list[dict[str, Any]]) -> bool:
    if not criteria.requires_run:
        return False
    return _last_failed_run_step_after_latest_change(observations) > _last_successful_run_step_after_latest_change(
        observations
    )


def _format_repair_guidance(criteria: CompletionCriteria, observations: list[dict[str, Any]]) -> str:
    if not _repair_required(criteria, observations):
        return ""

    failed = _last_failed_run_after_latest_change(observations)
    output = _truncate(_one_line(_format_shell_result(failed.get("result", {}))), 900) if failed else ""
    lines = [
        "Repair required:",
        "- The last verification command failed after a code edit.",
        "- Do not answer or finish yet.",
        "- Choose a corrective local action: read_file, replace_in_file, write_file, or run_shell.",
        "- If the failing source has already been read, the next action should usually edit the file. Do not rerun the same failed command until a file edit or materially different command/stdin changes the outcome.",
        "- Use the traceback/output as ground truth. If the command failed due missing CLI arguments, rerun with representative arguments or edit the file to include a demo/test entry point.",
        "- For assertion failures in generated algorithm tests, audit the test with a brute-force oracle or validity helper before changing expected constants.",
        "- For returned-index problems, repair tests to check index validity and target satisfaction when multiple index orders or answers are possible.",
        "- Shell commands are non-interactive. If the program reads input(), pass stdin in run_shell or edit the code to avoid interactive input during verification.",
    ]
    if output:
        lines.append(f"- Last failure: {output}")
    return "\n".join(lines)


def _completion_failure_observation(
    step: int,
    action: dict[str, Any],
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> dict[str, Any]:
    missing = _combined_completion_missing(criteria, observations, contract)
    return {
        "step": step,
        "action": _public_action(action),
        "signature": _action_signature(action),
        "result": {"ok": False, "error": "Completion criteria not met. " + " ".join(missing)},
    }


def _should_reject_completion_action(
    kind: str,
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> bool:
    if kind not in {"answer", "finish"}:
        return False
    if _repair_required(criteria, observations):
        return True
    if not _combined_completion_missing(criteria, observations, contract):
        return False
    if kind == "finish":
        return True
    return _has_started_local_work(observations)


def _should_finish_with_assistant_response(
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> bool:
    if contract is None or not contract.has_obligation("assistant_response") or not observations:
        return False
    return not _combined_completion_missing(criteria, observations, contract)


def _combined_completion_missing(
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> list[str]:
    missing = [*_completion_missing(criteria, observations)]
    if contract is not None:
        missing.extend(task_contract_missing(contract, observations))
    return _dedupe_strings(missing)


def _contract_obligation_by_id(contract: TaskContract, obligation_id: str) -> Any | None:
    for obligation in contract.obligations:
        if obligation.id == obligation_id:
            return obligation
    return None


def _contract_observation_steps(obligation: Any, observations: list[dict[str, Any]]) -> list[int]:
    steps: list[int] = []
    latest_change = _last_successful_change_step(observations)
    for observation in observations:
        if obligation.kind == "source_report" and int(observation.get("step", 0)) <= latest_change:
            continue
        if _observation_satisfies_contract_obligation(observation, obligation):
            steps.append(int(observation.get("step", 0)))
    return steps


def _has_missing_workspace_change(contract: TaskContract, observations: list[dict[str, Any]]) -> bool:
    return any(
        obligation.kind == "workspace_change"
        and obligation.required
        and not _contract_observation_steps(obligation, observations)
        for obligation in contract.obligations
    )


def _must_run_before_workspace_change(contract: TaskContract, run_obligation: Any) -> bool:
    for constraint in contract.constraints:
        if constraint.kind != "before" or constraint.first != run_obligation.id:
            continue
        second = _contract_obligation_by_id(contract, constraint.second)
        if second is not None and second.kind == "workspace_change":
            return True
    return False


def _observation_satisfies_contract_obligation(observation: dict[str, Any], obligation: Any) -> bool:
    result = observation.get("result", {})
    if not result.get("ok"):
        return False
    action = observation.get("action", {})
    action_name = action.get("action")
    target_path = _contract_target_path(obligation)

    if obligation.kind == "workspace_change":
        return action_name in {"write_file", "replace_in_file", "edit_repair"} and _result_path_matches(result, target_path)
    if obligation.kind == "workspace_delete":
        return action_name == "delete_file" and _result_path_matches(result, target_path)
    if obligation.kind in {"source_inspection", "source_report"}:
        return action_name == "read_file" and _result_path_matches(result, target_path)
    if obligation.kind == "edit_review":
        return (
            action_name == "edit_review"
            and bool(result.get("satisfied"))
            and _result_path_matches(result, target_path)
        )
    if obligation.kind == "local_execution":
        if action_name != "run_shell" or not _is_executed_shell_observation(observation):
            return False
        if target_path and target_path not in _command_path_tokens(str(action.get("command", ""))):
            return False
        expected = str(obligation.params.get("expected_text") or "").strip()
        return not expected or expected in _shell_output_text(observation)
    if obligation.kind == "visible_output":
        return action_name == "run_shell" and _has_meaningful_shell_output(observation)
    if obligation.kind == "test_evidence":
        return action_name == "run_shell" and _has_passing_test_evidence(observation)
    return False


def _validate_action_against_contract(
    action: dict[str, Any],
    contract: TaskContract,
    observations: list[dict[str, Any]],
) -> str | None:
    pending_change = _pending_workspace_change(contract, observations)
    if pending_change is None:
        return None

    kind = str(action.get("action", "")).strip()
    target_path = _contract_target_path(pending_change)
    if kind in {"write_file", "replace_in_file"}:
        action_path = str(action.get("path", "")).replace("\\", "/").lstrip("./")
        if target_path and action_path != target_path:
            return f"The outstanding workspace change targets {target_path}; edit that path."
        return None
    if kind in {"read_file", "list_files", "search_text"} and not _has_read_contract_target(
        pending_change, observations
    ):
        return None
    return (
        "The task contract still requires a workspace change. "
        "Choose write_file or replace_in_file for the target instead of answering, finishing, or running."
    )


def _tool_intent_for_action(intent: str, action: dict[str, Any], contract: TaskContract) -> str:
    if action.get("action") not in {"write_file", "replace_in_file", "delete_file"}:
        return intent
    if any(
        obligation.required and obligation.kind in {"workspace_change", "workspace_delete"}
        for obligation in contract.obligations
    ):
        return "edit"
    return intent


def _has_read_contract_target(obligation: Any, observations: list[dict[str, Any]]) -> bool:
    target_path = _contract_target_path(obligation)
    return any(
        observation.get("result", {}).get("ok")
        and observation.get("action", {}).get("action") == "read_file"
        and _result_path_matches(observation.get("result", {}), target_path)
        for observation in observations
    )


def _pending_workspace_change(contract: TaskContract, observations: list[dict[str, Any]]) -> Any | None:
    return next(
        (
            obligation
            for obligation in contract.obligations
            if obligation.kind == "workspace_change"
            and obligation.required
            and not _contract_observation_steps(obligation, observations)
        ),
        None,
    )


def _format_next_action_constraint(contract: TaskContract, observations: list[dict[str, Any]]) -> str:
    pending_change = _pending_workspace_change(contract, observations)
    if pending_change is None or not _has_read_contract_target(pending_change, observations):
        return ""
    target_path = _contract_target_path(pending_change) or "the requested target"
    return (
        "Next action constraint:\n"
        f"- The required workspace change for {target_path} is still missing.\n"
        "- The target has already been inspected. Return write_file or replace_in_file now."
    )


def _has_unsatisfied_recent_run_for_obligation(obligation: Any, observations: list[dict[str, Any]]) -> bool:
    expected = str(obligation.params.get("expected_text") or "").strip()
    target_path = _contract_target_path(obligation)
    if not expected or not target_path:
        return False
    latest_change = _last_successful_change_step(observations)
    for observation in observations:
        if int(observation.get("step", 0)) <= latest_change:
            continue
        action = observation.get("action", {})
        if action.get("action") != "run_shell":
            continue
        if target_path not in _command_path_tokens(str(action.get("command", ""))):
            continue
        if observation.get("result", {}).get("ok") and expected not in _shell_output_text(observation):
            return True
    return False


def _is_contract_repeated_unsatisfied_run(
    action: dict[str, Any],
    contract: TaskContract,
    observations: list[dict[str, Any]],
) -> bool:
    if action.get("action") != "run_shell":
        return False
    command = _command_signature(str(action.get("command", "")))
    if not command:
        return False
    latest_change = _last_successful_change_step(observations)
    current_paths = _command_path_tokens(str(action.get("command", "")))
    for observation in observations:
        if int(observation.get("step", 0)) <= latest_change:
            continue
        prior_action = observation.get("action", {})
        if prior_action.get("action") != "run_shell":
            continue
        if _command_signature(str(prior_action.get("command", ""))) != command:
            continue
        if not observation.get("result", {}).get("ok"):
            continue
        for obligation in contract.obligations:
            if obligation.kind != "local_execution":
                continue
            expected = str(obligation.params.get("expected_text") or "").strip()
            target_path = _contract_target_path(obligation)
            if not expected or not target_path or target_path not in current_paths:
                continue
            if expected not in _shell_output_text(observation):
                return True
    return False


def _contract_target_path(obligation: Any) -> str:
    return str(obligation.params.get("target_path") or "").replace("\\", "/").lstrip("./")


def _contract_edit_scope(contract: TaskContract) -> str | None:
    for obligation in contract.obligations:
        if obligation.kind != "workspace_change":
            continue
        scope = str(obligation.params.get("edit_scope") or "").strip()
        if scope in {"tests_only", "implementation", "whole_file"}:
            return scope
    return None


def _deterministic_edit_scope_issues(
    path: str,
    edit_scope: str | None,
    original_source: str,
    edited_source: str,
) -> list[str]:
    if edit_scope != "tests_only" or not path.endswith(".py") or not original_source:
        return []
    try:
        original_tree = ast.parse(original_source)
        edited_tree = ast.parse(edited_source)
    except SyntaxError as exc:
        return [f"The edited Python file is not syntactically valid: {exc.msg} at line {exc.lineno}."]

    original_symbols = _protected_python_symbols(original_tree)
    edited_symbols = _protected_python_symbols(edited_tree)
    issues: list[str] = []
    for name, original_node in original_symbols.items():
        edited_node = edited_symbols.get(name)
        if edited_node is None:
            issues.append(f"Preserve the existing implementation symbol `{name}` while changing only tests.")
            continue
        if ast.dump(original_node, include_attributes=False) != ast.dump(edited_node, include_attributes=False):
            issues.append(f"Do not modify implementation symbol `{name}` when the request is test-only.")
    return issues


def _review_failure_is_external_evidence_only(reason: str, missing: list[str]) -> bool:
    complaints = missing or ([reason] if reason else [])
    return bool(complaints) and all(_is_external_evidence_complaint(item) for item in complaints)


def _is_external_evidence_complaint(text: str) -> bool:
    normalized = _one_line(text).lower()
    patterns = (
        r"\b(?:execution|runtime|command results?|stdout|stderr)\b",
        r"\b(?:tests?|program|script|command)\b.{0,50}\b(?:not\s+)?(?:run|ran|executed)\b",
        r"\b(?:results?|output)\b.{0,40}\b(?:display|displayed|shown|observed|provided|reported)\b",
        r"\b(?:passing|passed|success(?:ful)?)\b.{0,40}\b(?:evidence|results?|output)\b",
        r"\b(?:final\s+)?(?:code|source|file contents?)\b.{0,40}\b(?:display|displayed|shown|reported)\b",
        r"\b(?:display|show|report)\b.{0,40}\b(?:final\s+)?(?:code|source|file contents?)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _repair_python_assert_expected_literals(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    ]
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    except Exception:
        return None

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        comparison = node.test
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
            continue
        if len(comparison.comparators) != 1 or not isinstance(comparison.left, ast.Call):
            continue
        call = comparison.left
        if not isinstance(call.func, ast.Name) or call.func.id not in namespace:
            continue
        try:
            args = [ast.literal_eval(argument) for argument in call.args]
            kwargs = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in call.keywords
                if keyword.arg is not None
            }
            if len(kwargs) != len(call.keywords):
                continue
            expected = ast.literal_eval(comparison.comparators[0])
            actual = namespace[call.func.id](*args, **kwargs)
            ast.literal_eval(repr(actual))
        except Exception:
            continue
        if actual == expected:
            continue
        span = _ast_source_span(source, comparison.comparators[0])
        if span is not None:
            replacements.append((*span, repr(actual)))

    if not replacements:
        return None
    repaired = source
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = f"{repaired[:start]}{replacement}{repaired[end:]}"
    return repaired


def _ast_source_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    if not all(hasattr(node, field) for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
        return None
    lines = source.splitlines(keepends=True)
    start_line = int(getattr(node, "lineno")) - 1
    end_line = int(getattr(node, "end_lineno")) - 1
    if start_line < 0 or end_line >= len(lines):
        return None

    def character_column(line: str, byte_column: int) -> int:
        return len(line.encode("utf-8")[:byte_column].decode("utf-8", errors="ignore"))

    start = sum(len(line) for line in lines[:start_line]) + character_column(
        lines[start_line], int(getattr(node, "col_offset"))
    )
    end = sum(len(line) for line in lines[:end_line]) + character_column(
        lines[end_line], int(getattr(node, "end_col_offset"))
    )
    return start, end


def _protected_python_symbols(tree: ast.Module) -> dict[str, ast.AST]:
    protected: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                protected[node.name] = node
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                protected[node.name] = node
    return protected


def _result_path_matches(result: dict[str, Any], target_path: str) -> bool:
    if not target_path:
        return True
    return str(result.get("path") or "").replace("\\", "/").lstrip("./") == target_path


def _command_path_tokens(command: str) -> set[str]:
    return {
        path.replace("\\", "/").lstrip("./")
        for path in command_target_paths(command)
        if path.endswith(".py")
    }


def _should_stop_after_contract_success(
    intent: str,
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> bool:
    if contract is None or contract.has_obligation("assistant_response") or not observations:
        return False
    if intent not in {"read", "edit", "shell"}:
        return False
    latest_action = observations[-1].get("action", {}).get("action")
    if latest_action in {"write_file", "replace_in_file", "edit_repair"}:
        return False
    if _repair_required(criteria, observations):
        return False
    if _combined_completion_missing(criteria, observations, contract):
        return False
    return any(obligation.required for obligation in contract.obligations)


def _has_started_local_work(observations: list[dict[str, Any]]) -> bool:
    return any(
        observation.get("action", {}).get("action")
        in {"write_file", "replace_in_file", "edit_repair", "delete_file", "run_shell", "read_file", "search_text", "list_files"}
        for observation in observations
    )


def _has_failed_observation(observations: list[dict[str, Any]]) -> bool:
    return any(not observation.get("result", {}).get("ok", True) for observation in observations)


def _successful_runs_after_latest_change(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    change_step = _last_successful_change_step(observations)
    return [
        observation
        for observation in observations
        if int(observation.get("step", 0)) > change_step
        and observation.get("action", {}).get("action") == "run_shell"
        and _is_executed_shell_observation(observation)
        and observation.get("result", {}).get("ok")
    ]


def _last_failed_run_after_latest_change(observations: list[dict[str, Any]]) -> dict[str, Any]:
    change_step = _last_successful_change_step(observations)
    failed_runs = [
        observation
        for observation in observations
        if int(observation.get("step", 0)) > change_step
        and observation.get("action", {}).get("action") == "run_shell"
        and _is_executed_shell_observation(observation)
        and not observation.get("result", {}).get("ok")
    ]
    return failed_runs[-1] if failed_runs else {}


def _last_failed_run_step_after_latest_change(observations: list[dict[str, Any]]) -> int:
    failed = _last_failed_run_after_latest_change(observations)
    return int(failed.get("step", 0)) if failed else 0


def _latest_failed_edit_review_after_latest_change(observations: list[dict[str, Any]]) -> dict[str, Any]:
    latest_change = _last_successful_change_step(observations)
    failed = [
        observation
        for observation in observations
        if int(observation.get("step", 0)) > latest_change
        and observation.get("action", {}).get("action") == "edit_review"
        and not observation.get("result", {}).get("ok")
    ]
    return failed[-1] if failed else {}


def _last_successful_run_step_after_latest_change(observations: list[dict[str, Any]]) -> int:
    runs = _successful_runs_after_latest_change(observations)
    return max((int(observation.get("step", 0)) for observation in runs), default=0)


def _has_test_evidence(observation: dict[str, Any]) -> bool:
    command = str(observation.get("action", {}).get("command", "")).lower()
    output = _shell_output_text(observation).lower()
    if "unittest" in command or "pytest" in command:
        return True
    return bool(
        re.search(r"\bran\s+\d+\s+tests?\b", output)
        or re.search(r"\b\d+\s+passed\b", output)
        or re.search(r"\ball\s+tests?\s+passed\b", output)
        or re.search(r"\btest\s+cases?\s+\d+\s+passed\b", output)
        or re.search(r"\bok\b", output)
    )


def _is_executed_shell_observation(observation: dict[str, Any]) -> bool:
    if observation.get("action", {}).get("action") != "run_shell":
        return False
    result = observation.get("result", {})
    return any(key in result for key in {"returncode", "stdout", "stderr"})


def _has_passing_test_evidence(observation: dict[str, Any]) -> bool:
    return _has_test_evidence(observation) and not _has_failing_test_output(observation)


def _has_failing_test_output(observation: dict[str, Any]) -> bool:
    output = _shell_output_text(observation)
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


def _validate_action(
    action: dict[str, Any],
    intent: str,
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    workspace: Path | None = None,
) -> str | None:
    kind = str(action.get("action", "")).strip()
    allowed = {
        "answer",
        "delete_file",
        "finish",
        "list_files",
        "read_file",
        "replace_in_file",
        "run_shell",
        "search_text",
        "write_file",
    }
    if kind not in allowed:
        return f"Unknown action '{kind}'."
    if kind == "write_file":
        if intent != "edit":
            return "write_file is only valid for edit requests."
        if not str(action.get("path", "")).strip():
            return "write_file requires a non-empty path."
        if "content" not in action:
            return "write_file requires content."
    elif kind == "replace_in_file":
        if intent != "edit":
            return "replace_in_file is only valid for edit requests."
        if not str(action.get("path", "")).strip():
            return "replace_in_file requires a non-empty path."
        if not str(action.get("old", "")):
            return "replace_in_file requires non-empty old text."
        if "new" not in action:
            return "replace_in_file requires new text."
    elif kind == "delete_file":
        if intent != "edit":
            return "delete_file is only valid for edit requests."
        if not str(action.get("path", "")).strip():
            return "delete_file requires a non-empty path."
    elif kind == "run_shell":
        if intent not in {"edit", "shell"}:
            return "run_shell is only valid for edit or shell requests."
        if not str(action.get("command", "")).strip():
            return "run_shell requires a non-empty command."
        stdin = action.get("stdin")
        if stdin is not None and not isinstance(stdin, str):
            return "run_shell stdin must be a string when provided."
        missing_start_dir = _missing_unittest_start_dir(action, workspace)
        if missing_start_dir:
            return (
                f"run_shell cannot use unittest discovery because the start directory does not exist: "
                f"{missing_start_dir}. Create a test file under that directory or run the target script directly."
            )
        if _run_shell_lacks_requested_output(action, criteria, workspace):
            return (
                "This run_shell command cannot satisfy the requested visible output because the target Python "
                "file has no print/test/stdout entry point. Edit the file to display results, create tests, "
                "or run a different command that produces output."
            )
    elif kind == "read_file" and not str(action.get("path", "")).strip():
        return "read_file requires a non-empty path."
    elif kind == "search_text" and not str(action.get("pattern", "")).strip():
        return "search_text requires a non-empty pattern."
    elif kind in {"answer", "finish"}:
        if kind == "finish" and intent == "edit" and not observations:
            return "finish is only valid after completing the requested work."
        if not str(action.get("message", "")).strip():
            return f"{kind} requires a non-empty message."
        if _repair_required(criteria, observations):
            return (
                f"{kind} is not valid while repair is required after a failed run. "
                "Choose read_file, replace_in_file, write_file, or run_shell to fix or rerun the local code."
            )
        if _should_reject_completion_action(kind, criteria, observations):
            return (
                f"{kind} is not valid because completion requirements are not met. "
                "Choose a local action that produces the missing evidence."
            )
    return None


def _missing_unittest_start_dir(action: dict[str, Any], workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    start_dir = _unittest_discover_start_dir(str(action.get("command", "")).strip())
    if start_dir is None:
        return None
    start_path = (workspace / start_dir).resolve()
    try:
        if not start_path.is_relative_to(workspace.resolve()):
            return None
    except ValueError:
        return None
    return start_dir if not start_path.exists() else None


def _unittest_discover_start_dir(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if len(tokens) < 4 or Path(tokens[0]).name not in {"python", "python3"}:
        return None
    if tokens[1:4] != ["-m", "unittest", "discover"]:
        return None
    start_dir = "."
    for index, token in enumerate(tokens):
        if token in {"-s", "--start-directory"} and index + 1 < len(tokens):
            start_dir = tokens[index + 1]
    return start_dir


def _run_shell_lacks_requested_output(
    action: dict[str, Any],
    criteria: CompletionCriteria,
    workspace: Path | None,
) -> bool:
    if not criteria.requires_output or workspace is None:
        return False
    command = str(action.get("command", "")).strip()
    path = _single_python_file_command_path(command)
    if path is None or _is_python_test_file(path):
        return False
    file_path = (workspace / path).resolve()
    try:
        if not file_path.is_relative_to(workspace.resolve()):
            return False
    except ValueError:
        return False
    return file_path.exists() and not _python_file_has_visible_output(file_path)


def _single_python_file_command_path(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if len(tokens) < 2 or Path(tokens[0]).name not in {"python", "python3"}:
        return None
    if tokens[1].startswith("-") or not tokens[1].endswith(".py"):
        return None
    return tokens[1]


def _salvage_action_from_text(
    task: str,
    intent: str,
    text: str,
    task_spec: TaskSpec | None = None,
) -> dict[str, Any] | None:
    if intent != "edit":
        return None
    path = task_spec.target_path if task_spec and task_spec.target_path else _filename_from_task(task)
    if not path:
        return None
    code = _python_code_block(text)
    if code is None:
        return None
    return {"action": "write_file", "path": path, "content": code.rstrip() + "\n"}


def _filename_from_task(task: str) -> str | None:
    path = _explicit_filename_from_task(task) or _inferred_python_filename(task)
    if not path:
        return None
    if path.startswith("/") or ".." in Path(path).parts:
        return None
    return path


def _explicit_filename_from_task(task: str) -> str | None:
    match = re.search(r"\b[\w./-]+\.py\b", task)
    if not match:
        return None
    path = match.group(0).strip("./")
    if path.startswith("/") or ".." in Path(path).parts:
        return None
    return path


def _inferred_python_filename(task: str) -> str | None:
    normalized = _lookup_text(task)
    problem_names = {
        "3sum": "three_sum.py",
        "3 sum": "three_sum.py",
        "three sum": "three_sum.py",
        "2sum": "two_sum.py",
        "2 sum": "two_sum.py",
        "two sum": "two_sum.py",
        "palindrome": "palindrome.py",
        "fibonacci": "fibonacci.py",
    }
    for phrase, filename in problem_names.items():
        if phrase in normalized:
            return filename
    if not _looks_like_python_artifact_request(task):
        return None
    slug = _python_filename_slug_from_task(normalized)
    if not slug:
        return None
    return f"{slug}.py"


def _looks_like_python_artifact_request(task: str) -> bool:
    text = task.lower()
    return bool(
        ".py" in text
        or re.search(r"\bpython\b", text)
        or re.search(r"\b(script|program)\b", text)
    )


def _python_filename_slug_from_task(normalized_task: str) -> str | None:
    patterns = [
        r"\bprints?\s+(?:out\s+)?(.+?)(?:\s+and\b|$)",
        r"\boutputs?\s+(.+?)(?:\s+given\b|\s+with\b|\s+for\b|\s+and\b|$)",
        r"\bimplements?\s+(?:the\s+)?(.+?)(?:\s+problem\b|\s+solution\b|\s+algorithm\b|\s+and\b|$)",
        r"\bsolves?\s+(?:the\s+)?(.+?)(?:\s+problem\b|\s+solution\b|\s+algorithm\b|\s+and\b|$)",
        r"\bfor\s+(?:the\s+)?(.+?)(?:\s+problem\b|\s+and\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_task)
        if match:
            slug = _slugify_filename_phrase(match.group(1))
            if slug:
                return slug

    return _slugify_filename_phrase(normalized_task)


def _slugify_filename_phrase(phrase: str) -> str | None:
    stop_words = {
        "a",
        "an",
        "and",
        "as",
        "build",
        "code",
        "create",
        "file",
        "generate",
        "implement",
        "implements",
        "in",
        "new",
        "out",
        "output",
        "outputs",
        "print",
        "prints",
        "program",
        "python",
        "script",
        "simply",
        "small",
        "solve",
        "solves",
        "that",
        "the",
        "to",
        "write",
    }
    tokens = [token for token in phrase.split() if token not in stop_words]
    while tokens and tokens[-1] in {"example", "problem", "solution"}:
        tokens.pop()
    if not tokens or not any(re.search(r"[a-z]", token) for token in tokens):
        return None
    if len(tokens) > 5:
        tokens = tokens[:5]
    slug = "_".join(tokens)
    if slug[0].isdigit():
        slug = f"task_{slug}"
    return slug


def _python_code_block(text: str) -> str | None:
    for language, content in _fenced_blocks(text, with_language=True):
        if language in {"python", "py"}:
            return content
    for language, content in _fenced_blocks(text, with_language=True):
        if language == "" and _looks_like_python(content):
            return content
    return None


def _normalize_written_content(content: str) -> str:
    normalized = _strip_outer_code_fence(content)
    if normalized.count("\\n") >= 3 and normalized.count("\n") <= 1:
        normalized = normalized.replace("\\n", "\n").replace("\\t", "\t")
        normalized = _strip_outer_code_fence(normalized)
    return normalized


def _strip_outer_code_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if not match:
        return content
    return match.group(1).strip() + "\n"


def _looks_like_python(text: str) -> bool:
    return bool(re.search(r"^\s*(def|class|import|from|assert|print)\b", text, re.MULTILINE))


def _protocol_error(reason: str, content: str) -> str:
    return (
        "Protocol correction:\n"
        f"- The previous response was invalid: {reason}\n"
        "- Return exactly one valid JSON action object. No markdown. No prose.\n"
        f"- Previous response excerpt: {_truncate(_one_line(content), 700)}"
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[middle omitted]...\n"
    available = max(0, limit - len(marker))
    head_length = int(available * 0.6)
    tail_length = available - head_length
    return f"{text[:head_length]}{marker}{text[-tail_length:] if tail_length else ''}"


def _fenced_blocks(text: str, *, with_language: bool = False):
    blocks = []
    for match in re.finditer(r"```([a-zA-Z0-9_-]*)\s*(.*?)```", text, re.DOTALL):
        language = match.group(1).strip().lower()
        content = match.group(2).strip()
        blocks.append((language, content) if with_language else content)
    return blocks


def _format_json(title: str, value: Any) -> str:
    return f"{title}:\n{json.dumps(value, indent=2, ensure_ascii=False)}"


def _format_edit_review_history(observations: list[dict[str, Any]]) -> str:
    compact: list[dict[str, Any]] = []
    for observation in observations[-12:]:
        action = observation.get("action", {})
        result = observation.get("result", {})
        compact.append(
            {
                "step": observation.get("step"),
                "action": action.get("action"),
                "path": result.get("path") or action.get("path"),
                "command": action.get("command"),
                "ok": result.get("ok"),
                "error": _truncate(_one_line(str(result.get("error", ""))), 400),
                "reason": _truncate(_one_line(str(result.get("reason", ""))), 400),
                "missing": result.get("missing", []),
            }
        )
    return _format_json("Relevant action history", compact)


def _format_latest_failed_edit_review(observations: list[dict[str, Any]]) -> str:
    failed = _latest_failed_edit_review_after_latest_change(observations)
    if not failed:
        return "No failed edit review is available."
    result = failed.get("result", {})
    lines = []
    reason = _one_line(str(result.get("reason", "")))
    if reason:
        lines.append(f"Reason: {reason}")
    missing = result.get("missing", [])
    if isinstance(missing, list) and missing:
        lines.append("Missing:")
        lines.extend(f"- {_one_line(str(item))}" for item in missing if str(item).strip())
    error = _one_line(str(result.get("error", "")))
    if error:
        lines.append(f"Error: {error}")
    return "\n".join(lines) if lines else "The previous edit review failed without details."


def _format_agent_state(agent: LocalAgent) -> str:
    state: dict[str, Any] = {
        "last_written_file": agent.last_written_file,
        "last_shell_command": agent.last_shell_command,
    }
    if agent.last_shell_result is not None:
        state["last_shell_result"] = {
            "ok": agent.last_shell_result.get("ok"),
            "returncode": agent.last_shell_result.get("returncode"),
            "stdout": _truncate(str(agent.last_shell_result.get("stdout", "")), 500),
            "stderr": _truncate(str(agent.last_shell_result.get("stderr", "")), 500),
        }
    return _format_json("Agent state", state)


def _public_action(action: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in action.items() if key not in {"content", "stdin"}}
    if "content" in action:
        public["content_bytes"] = len(str(action["content"]).encode("utf-8"))
    if "stdin" in action:
        public["stdin_bytes"] = len(str(action["stdin"]).encode("utf-8"))
    return public


def _action_signature(action: dict[str, Any]) -> str:
    kind = str(action.get("action", ""))
    if kind == "write_file":
        content_hash = hashlib.sha256(str(action.get("content", "")).encode("utf-8")).hexdigest()
        return f"write_file:{action.get('path', '')}:{content_hash}"
    if kind == "replace_in_file":
        old_hash = hashlib.sha256(str(action.get("old", "")).encode("utf-8")).hexdigest()
        new_hash = hashlib.sha256(str(action.get("new", "")).encode("utf-8")).hexdigest()
        return f"replace_in_file:{action.get('path', '')}:{old_hash}:{new_hash}"
    if kind == "run_shell":
        return f"run_shell:{_command_signature(str(action.get('command', '')))}"
    return json.dumps(_public_action(action), sort_keys=True, ensure_ascii=False)


def _is_repeated_write(action: dict[str, Any], observations: list[dict[str, Any]]) -> bool:
    if action.get("action") not in {"write_file", "replace_in_file"}:
        return False
    signature = _action_signature(action)
    return any(observation.get("signature") == signature for observation in observations)


def _is_repeated_unproductive_run(
    action: dict[str, Any],
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
) -> bool:
    if action.get("action") != "run_shell" or not criteria.requires_output:
        return False
    command = _command_signature(str(action.get("command", "")))
    if not command:
        return False
    latest_change_step = _last_successful_change_step(observations)
    for observation in observations:
        if int(observation.get("step", 0)) <= latest_change_step:
            continue
        if observation.get("action", {}).get("action") != "run_shell":
            continue
        if _command_signature(str(observation.get("action", {}).get("command", ""))) != command:
            continue
        if not _is_executed_shell_observation(observation):
            continue
        if observation.get("result", {}).get("ok") and not _has_meaningful_shell_output(observation):
            return True
    return False


def _is_repeated_failed_run(action: dict[str, Any], observations: list[dict[str, Any]]) -> bool:
    if action.get("action") != "run_shell":
        return False
    command = _command_signature(str(action.get("command", "")))
    if not command:
        return False
    latest_change_step = _last_successful_change_step(observations)
    return any(
        int(observation.get("step", 0)) > latest_change_step
        and observation.get("action", {}).get("action") == "run_shell"
        and _command_signature(str(observation.get("action", {}).get("command", ""))) == command
        and _is_executed_shell_observation(observation)
        and not observation.get("result", {}).get("ok")
        for observation in observations
    )


def _last_action_step(observations: list[dict[str, Any]], action_name: str) -> int:
    return max(
        (
            int(observation.get("step", 0))
            for observation in observations
            if observation.get("action", {}).get("action") == action_name
        ),
        default=0,
    )


def _has_run_command_after_step(command: str, step: int, observations: list[dict[str, Any]]) -> bool:
    signature = _command_signature(command)
    return any(
        int(observation.get("step", 0)) > step
        and observation.get("action", {}).get("action") == "run_shell"
        and _command_signature(str(observation.get("action", {}).get("command", ""))) == signature
        and _is_executed_shell_observation(observation)
        for observation in observations
    )


def _command_signature(command: str) -> str:
    command = command.strip()
    if not command:
        return ""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return " ".join(command.split())
    if not tokens:
        return ""
    if Path(tokens[0]).name in {"python", "python3"}:
        index = 1
        while index < len(tokens) and tokens[index] in {"-u"}:
            index += 1
        if index < len(tokens) and tokens[index].endswith(".py"):
            script = str(PurePosixPath(tokens[index]).as_posix()).lstrip("./")
            return " ".join(["python", script, *tokens[index + 1 :]])
    return " ".join(tokens)


def _last_successful_change_step(observations: list[dict[str, Any]]) -> int:
    return max(
        (
            int(observation.get("step", 0))
            for observation in observations
            if observation.get("action", {}).get("action") in {"write_file", "replace_in_file", "edit_repair"}
            and observation.get("result", {}).get("ok")
        ),
        default=0,
    )


def _source_before_latest_change(path: str, observations: list[dict[str, Any]]) -> str:
    normalized_path = path.replace("\\", "/").lstrip("./")
    latest_change = _last_successful_change_step(observations)
    reads = [
        observation
        for observation in observations
        if int(observation.get("step", 0)) < latest_change
        and observation.get("action", {}).get("action") == "read_file"
        and str(observation.get("result", {}).get("path", "")).replace("\\", "/").lstrip("./") == normalized_path
        and observation.get("result", {}).get("ok")
    ]
    if not reads:
        return ""
    result = reads[-1].get("result", {})
    return str(result.get("raw_content") or result.get("content") or "")


def _should_stop_after_repair_stall(criteria: CompletionCriteria, observations: list[dict[str, Any]]) -> bool:
    if not _repair_required(criteria, observations):
        return False
    failed_step = _last_failed_run_step_after_latest_change(observations)
    stalled = [
        observation
        for observation in observations
        if int(observation.get("step", 0)) > failed_step
        and not observation.get("result", {}).get("ok")
        and observation.get("action", {}).get("action") in {"answer", "finish", "write_file", "replace_in_file", "edit_repair", "run_shell"}
    ]
    return len(stalled) >= 2


def _format_repair_stall_reply(observations: list[dict[str, Any]]) -> str:
    return "Could not repair the failed verification after several recovery attempts.\n\n" + _format_observations(
        observations
    )


def _should_stop_after_failure(result: dict[str, Any]) -> bool:
    error = str(result.get("error", "")).lower()
    return "declined" in error or "blocked" in error or "escapes workspace" in error


def _is_user_declined_result(result: dict[str, Any]) -> bool:
    return "declined" in str(result.get("error", "")).lower()


def _drop_latest_user_message(messages: list[dict[str, Any]], task: str) -> None:
    if len(messages) <= 1:
        return
    latest = messages[-1]
    if latest.get("role") == "user" and latest.get("content") == task:
        messages.pop()


def _should_stop_after_success(
    intent: str,
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> bool:
    if contract is not None and contract.has_obligation("assistant_response"):
        return False
    has_execution_obligation = contract.has_obligation("local_execution") if contract is not None else False
    if intent not in {"edit", "shell"} or not (criteria.requires_run or has_execution_obligation) or not observations:
        return False
    latest = observations[-1]
    if latest.get("action", {}).get("action") != "run_shell":
        return False
    if not latest.get("result", {}).get("ok"):
        return False
    return not _combined_completion_missing(criteria, observations, contract)


def _should_stop_after_simple_edit(
    task: str,
    intent: str,
    criteria: CompletionCriteria,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> bool:
    if contract is not None and contract.has_obligation("assistant_response"):
        return False
    if intent != "edit" or criteria.requires_run or not observations:
        return False
    if _combined_completion_missing(criteria, observations, contract):
        return False
    if not _looks_like_simple_single_file_edit(task):
        return False
    if _successful_change_count(observations) != 1:
        return False
    latest = observations[-1]
    if latest.get("action", {}).get("action") not in {"write_file", "replace_in_file"}:
        return False
    return bool(latest.get("result", {}).get("ok") and latest.get("result", {}).get("path"))


def _successful_change_count(observations: list[dict[str, Any]]) -> int:
    return sum(
        1
        for observation in observations
            if observation.get("action", {}).get("action") in {"write_file", "replace_in_file", "edit_repair"}
        and observation.get("result", {}).get("ok")
    )


def _looks_like_simple_single_file_edit(task: str) -> bool:
    text = task.lower()
    if not any(marker in text for marker in {"file", "script", ".py"}):
        return False
    multi_step_markers = {
        " and ",
        " then ",
        " also ",
        "test",
        "tests",
        "unittest",
        "pytest",
        "run",
        "verify",
        "check",
        "fix",
        "debug",
        "refactor",
        "multiple",
        "several",
        "both",
    }
    if any(marker in text for marker in multi_step_markers):
        return False
    if re.search(r"\bfiles\b", text):
        return False
    if re.search(r"\b(two|three|four|five|\d+)\s+files?\b", text):
        return False
    return True


def _simple_edit_reply(observation: dict[str, Any]) -> str:
    action_name = observation.get("action", {}).get("action")
    path = observation.get("result", {}).get("path", "the requested file")
    verb = "Updated" if action_name == "replace_in_file" else "Created"
    return f"{verb} `{path}`."


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _python_run_command_for_path(path: str, source_path: Path | None = None) -> str:
    file_path = Path(path)
    if _is_python_test_file(path):
        return f"python3 -m unittest discover -s tests -p {shlex.quote(file_path.name)}"
    if source_path is not None and _python_has_unittest_cases(source_path):
        start_dir = str(file_path.parent) if str(file_path.parent) != "." else "."
        return (
            f"python3 -m unittest discover -s {shlex.quote(start_dir)} "
            f"-p {shlex.quote(file_path.name)}"
        )
    if source_path is not None and _python_has_uninvoked_test_functions(source_path):
        runner = (
            "import runpy,sys; "
            "ns=runpy.run_path(sys.argv[1]); "
            "tests=[value for name,value in ns.items() if name.startswith('test_') and callable(value)]; "
            "[test() for test in tests]; "
            "print(f'Ran {len(tests)} tests successfully.')"
        )
        return f"python3 -c {shlex.quote(runner)} {shlex.quote(path)}"
    return f"python3 {shlex.quote(path)}"


def _is_python_test_file(path: str) -> bool:
    file_path = Path(path)
    return (
        len(file_path.parts) >= 2
        and file_path.parts[0] == "tests"
        and file_path.name.startswith("test_")
        and file_path.suffix == ".py"
    )


def _python_has_uninvoked_test_functions(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    test_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
    if not test_names:
        return False
    main_calls: set[str] = set()
    for node in tree.body:
        if not _is_main_guard(node):
            continue
        for descendant in ast.walk(node):
            if isinstance(descendant, ast.Call) and isinstance(descendant.func, ast.Name):
                main_calls.add(descendant.func.id)
    return not test_names.issubset(main_calls)


def _python_has_unittest_cases(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "TestCase":
                return True
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "TestCase"
                and isinstance(base.value, ast.Name)
                and base.value.id == "unittest"
            ):
                return True
    return False


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
        return False
    values = [comparison.left, *comparison.comparators]
    return any(isinstance(value, ast.Name) and value.id == "__name__" for value in values) and any(
        isinstance(value, ast.Constant) and value.value == "__main__" for value in values
    )


def _python_file_has_visible_output(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return bool(
        re.search(r"\bprint\s*\(", content)
        or "unittest.main(" in content
        or "pytest.main(" in content
        or re.search(r"\bsys\.stdout\.write\s*\(", content)
    )


def _python_file_has_inline_tests(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(
        re.search(r"^\s*assert\s+", content, re.MULTILINE)
        or re.search(r"\bdef\s+test_[a-zA-Z0-9_]*\s*\(", content)
        or "unittest.main(" in content
        or "pytest.main(" in content
    )


def _python_file_requires_stdin(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(r"\binput\s*\(", content) or re.search(r"\bsys\.stdin\.", content))


def _format_final_reply(
    reply: str,
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> str:
    shell_results = _shell_results_for_final(observations, contract)
    contract_evidence = format_contract_evidence_for_final(contract, observations) if contract is not None else ""
    if not shell_results and not contract_evidence:
        return reply

    lines = [reply]
    if contract_evidence:
        lines.extend(["", contract_evidence])
    if shell_results:
        lines.extend(["", "Command results:"])
        for observation in shell_results:
            action = observation["action"]
            result = observation["result"]
            lines.append(f"$ {action.get('command', '')}")
            lines.append(_format_shell_result(result))
    return "\n".join(line for line in lines if line is not None).strip()


def _shell_results_for_final(
    observations: list[dict[str, Any]],
    contract: TaskContract | None = None,
) -> list[dict[str, Any]]:
    shell_results = [
        observation
        for observation in observations
        if observation.get("action", {}).get("action") == "run_shell"
    ]
    latest_change = _last_successful_change_step(observations)
    if not latest_change or _contract_has_pre_change_execution(contract):
        candidates = shell_results
    else:
        candidates = [
            observation
            for observation in shell_results
            if int(observation.get("step", 0)) > latest_change
        ] or shell_results
    successful = [
        observation
        for observation in candidates
        if observation.get("result", {}).get("ok") and _is_executed_shell_observation(observation)
    ]
    if not successful or _contract_has_pre_change_execution(contract):
        return candidates
    required_runs = _contract_required_run_count(contract)
    return successful[-max(1, required_runs) :]


def _contract_has_pre_change_execution(contract: TaskContract | None) -> bool:
    if contract is None:
        return False
    for constraint in contract.constraints:
        if constraint.kind != "before":
            continue
        first = _contract_obligation_by_id(contract, constraint.first)
        second = _contract_obligation_by_id(contract, constraint.second)
        if first is not None and second is not None:
            if first.kind == "local_execution" and second.kind == "workspace_change":
                return True
    return False


def _contract_required_run_count(contract: TaskContract | None) -> int:
    if contract is None:
        return 1
    return sum(
        int(obligation.params.get("min_successes", 1) or 1)
        for obligation in contract.obligations
        if obligation.kind == "local_execution" and obligation.required
    ) or 1


def _strip_trailing_question(text: str) -> str:
    stripped = text.strip()
    if not stripped.endswith("?"):
        return stripped
    boundary = max(stripped.rfind(".", 0, -1), stripped.rfind("!", 0, -1), stripped.rfind("\n", 0, -1))
    if boundary <= 0:
        return stripped
    question = stripped[boundary + 1 :].strip()
    if len(question) > 220 or not question.endswith("?"):
        return stripped
    return stripped[: boundary + 1].rstrip()


def _format_observations(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "No actions were completed."

    lines: list[str] = []
    for observation in observations:
        step = observation.get("step")
        action = observation.get("action", {})
        result = observation.get("result", {})
        name = action.get("action", "action")
        if result.get("ok"):
            detail = result.get("path") or result.get("returncode")
            lines.append(f"Step {step}: {name} completed" + (f" ({detail})" if detail is not None else "."))
            if name == "run_shell":
                output = _format_shell_result(result)
                if output:
                    lines.append(output)
        else:
            lines.append(f"Step {step}: {name} failed.")
            lines.append(_format_shell_result(result) if name == "run_shell" else str(result.get("error", "unknown error")))
    return "\n".join(lines).strip()


def _format_shell_result(result: dict[str, Any]) -> str:
    if not result.get("ok") and result.get("error"):
        return str(result["error"])
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()
    returncode = result.get("returncode")
    if stdout and stderr:
        return f"{stdout}\n\nstderr:\n{stderr}"
    if stdout:
        return stdout
    if stderr:
        return f"Command exited with {returncode}.\n{stderr}"
    return f"Command exited with {returncode}."


def _format_read_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return str(result.get("error", "Could not read file."))
    path = result.get("path", "file")
    content = str(result.get("content", "")).strip()
    if not content:
        return f"`{path}` is empty."
    return f"`{path}`:\n\n```text\n{content}\n```"
