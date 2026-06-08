import unittest

from local_agent.task_contract import (
    command_target_paths,
    contract_from_model_json,
    contract_missing,
    derive_task_contract,
    format_contract_evidence_for_final,
)


class TaskContractTests(unittest.TestCase):
    def test_unittest_discovery_command_resolves_test_file_target(self):
        paths = command_target_paths("python3 -m unittest discover -s tests -p test_sample.py")

        self.assertIn("tests/test_sample.py", paths)

    def test_contract_extracts_repeated_execution_obligation(self):
        contract = derive_task_contract(
            task="run helloworld.py 5 times",
            intent="shell",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
        )

        execution = next(obligation for obligation in contract.obligations if obligation.kind == "local_execution")

        self.assertEqual(execution.params["min_successes"], 5)

    def test_contract_counts_two_explicit_run_steps(self):
        contract = derive_task_contract(
            task="run app.py, change it, then run it again",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="app.py",
            operation="update",
        )

        execution = next(obligation for obligation in contract.obligations if obligation.kind == "local_execution")

        self.assertEqual(execution.params["min_successes"], 2)

    def test_run_again_after_edit_is_one_current_turn_execution(self):
        contract = derive_task_contract(
            task="change the test, run it again, and display results",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="app.py",
            operation="update",
        )

        execution = next(obligation for obligation in contract.obligations if obligation.kind == "local_execution")

        self.assertEqual(execution.params["min_successes"], 1)

    def test_test_only_edit_scope_is_preserved_in_contract(self):
        fallback = derive_task_contract(
            task="change the test in two_sum.py to a more complicated one",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="two_sum.py",
            operation="update",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "change_file",
                        "kind": "workspace_change",
                        "description": "Change two_sum.py.",
                        "required": True,
                        "params": {"target_path": "two_sum.py"},
                    }
                ],
                "constraints": [],
            },
            fallback=fallback,
            task="change the test in two_sum.py to a more complicated one",
        )
        change = next(obligation for obligation in contract.obligations if obligation.kind == "workspace_change")

        self.assertEqual(change.params["edit_scope"], "tests_only")

    def test_contract_missing_counts_successful_runs(self):
        contract = derive_task_contract(
            task="run helloworld.py 3 times",
            intent="shell",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "run_shell", "command": "python3 helloworld.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "hello\n", "stderr": ""},
            }
        ]

        missing = contract_missing(contract, observations)

        self.assertTrue(any("Only 1 of 3 requested successful local executions" in item for item in missing))

    def test_source_report_uses_read_file_evidence(self):
        contract = derive_task_contract(
            task="display the code in hello.py",
            intent="read",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="hello.py",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "read_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "content": "1: print('hello')"},
            }
        ]

        self.assertEqual(contract_missing(contract, observations), [])
        self.assertIn("1: print('hello')", format_contract_evidence_for_final(contract, observations))

    def test_source_report_after_edit_uses_latest_read_only(self):
        contract = derive_task_contract(
            task="change hello.py, display the code, and run it",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="hello.py",
            operation="update",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "read_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "content": "1: print('old')"},
            },
            {
                "step": 2,
                "action": {"action": "replace_in_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py"},
            },
            {
                "step": 3,
                "action": {"action": "edit_review", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "satisfied": True},
            },
        ]

        self.assertTrue(any("final requested source" in item for item in contract_missing(contract, observations)))

        observations.append(
            {
                "step": 4,
                "action": {"action": "read_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "content": "1: print('new')"},
            }
        )
        evidence = format_contract_evidence_for_final(contract, observations)

        self.assertIn("print('new')", evidence)
        self.assertNotIn("print('old')", evidence)

    def test_printing_program_does_not_imply_source_report(self):
        contract = derive_task_contract(
            task="write a python file that prints Hello World",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="hello_world.py",
            operation="create",
        )

        self.assertFalse(contract.has_obligation("source_report"))

    def test_displaying_command_results_does_not_imply_source_report(self):
        contract = derive_task_contract(
            task="run ordered_probe.py, change it, run it again, and display both command results",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="ordered_probe.py",
            operation="update",
        )

        self.assertFalse(contract.has_obligation("source_report"))

    def test_conversational_subtask_gets_fallback_obligation(self):
        contract = derive_task_contract(
            task="write and run hello.py, then tell me how you are feeling today",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="hello.py",
            operation="create",
        )

        self.assertTrue(contract.has_obligation("assistant_response"))

    def test_update_contract_requires_inspection_and_semantic_review(self):
        contract = derive_task_contract(
            task="modify pricing.py so tax is a percentage",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="pricing.py",
            operation="update",
        )

        self.assertTrue(contract.has_obligation("source_inspection"))
        self.assertTrue(contract.has_obligation("workspace_change"))
        self.assertTrue(contract.has_obligation("edit_review"))
        self.assertEqual(
            [(constraint.first, constraint.second) for constraint in contract.constraints],
            [("source_inspection", "workspace_change"), ("workspace_change", "edit_review")],
        )

    def test_update_contract_rejects_write_without_successful_review(self):
        contract = derive_task_contract(
            task="modify pricing.py so tax is a percentage",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="pricing.py",
            operation="update",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "read_file", "path": "pricing.py"},
                "result": {"ok": True, "path": "pricing.py", "content": "1: return price + tax"},
            },
            {
                "step": 2,
                "action": {"action": "replace_in_file", "path": "pricing.py"},
                "result": {"ok": True, "path": "pricing.py"},
            },
            {
                "step": 3,
                "action": {"action": "edit_review", "path": "pricing.py"},
                "result": {"ok": False, "path": "pricing.py", "satisfied": False},
            },
        ]

        missing = contract_missing(contract, observations)

        self.assertTrue(any("has not passed review" in item for item in missing))

        observations[-1]["result"] = {"ok": True, "path": "pricing.py", "satisfied": True}
        self.assertEqual(contract_missing(contract, observations), [])

    def test_model_contract_validates_ordered_run_edit_run_delete(self):
        fallback = derive_task_contract(
            task="run it, change it, run it again, then delete it",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="hello.py",
            operation="update",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "run_before",
                        "kind": "local_execution",
                        "description": "Run before editing.",
                        "required": True,
                        "params": {"target_path": "hello.py"},
                    },
                    {
                        "id": "change",
                        "kind": "workspace_change",
                        "description": "Change the file.",
                        "required": True,
                        "params": {"target_path": "hello.py"},
                    },
                    {
                        "id": "run_after",
                        "kind": "local_execution",
                        "description": "Run after editing.",
                        "required": True,
                        "params": {"target_path": "hello.py"},
                    },
                    {
                        "id": "delete",
                        "kind": "workspace_delete",
                        "description": "Delete the file.",
                        "required": True,
                        "params": {"target_path": "hello.py"},
                    },
                ],
                "constraints": [
                    {"kind": "before", "first": "run_before", "second": "change"},
                    {"kind": "before", "first": "change", "second": "run_after"},
                    {"kind": "before", "first": "run_after", "second": "delete"},
                ],
            },
            fallback=fallback,
        )

        incomplete = [
            {
                "step": 1,
                "action": {"action": "read_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "content": "1: print('old')"},
            },
            {
                "step": 2,
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "old\n", "stderr": ""},
            },
            {
                "step": 3,
                "action": {"action": "replace_in_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py"},
            },
        ]
        complete = [
            *incomplete,
            {
                "step": 4,
                "action": {"action": "edit_review", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py", "satisfied": True},
            },
            {
                "step": 5,
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "new\n", "stderr": ""},
            },
            {
                "step": 6,
                "action": {"action": "delete_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py"},
            },
        ]

        self.assertTrue(contract_missing(contract, incomplete))
        self.assertEqual(contract_missing(contract, complete), [])

    def test_model_assistant_response_is_filtered_when_not_requested(self):
        fallback = derive_task_contract(
            task="run hello.py 3 times",
            intent="shell",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="hello.py",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "run_program",
                        "kind": "local_execution",
                        "description": "Run the program.",
                        "required": True,
                        "params": {"target_path": "hello.py", "min_successes": 3},
                    },
                    {
                        "id": "respond",
                        "kind": "assistant_response",
                        "description": "Respond to the user.",
                        "required": True,
                    },
                ],
                "constraints": [],
            },
            fallback=fallback,
            task="run hello.py 3 times",
        )

        self.assertFalse(contract.has_obligation("assistant_response"))

    def test_model_execution_obligation_is_filtered_when_user_did_not_request_run(self):
        fallback = derive_task_contract(
            task="modify cache.py so it does not store None results",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="cache.py",
            operation="update",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "change_cache",
                        "kind": "workspace_change",
                        "description": "Change cache behavior.",
                        "required": True,
                        "params": {"target_path": "cache.py"},
                    },
                    {
                        "id": "run_tests",
                        "kind": "local_execution",
                        "description": "Run tests.",
                        "required": True,
                        "params": {"target_path": "cache.py"},
                    },
                ],
                "constraints": [],
            },
            fallback=fallback,
            task="modify cache.py so it does not store None results",
        )

        self.assertFalse(contract.has_obligation("local_execution"))
        self.assertTrue(contract.has_obligation("workspace_change"))

    def test_model_extra_execution_obligation_is_capped_to_current_request(self):
        fallback = derive_task_contract(
            task="change app.py, run it again, and display results",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="app.py",
            operation="update",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "run_before",
                        "kind": "local_execution",
                        "description": "Run before editing.",
                        "required": True,
                        "params": {"target_path": "app.py"},
                    },
                    {
                        "id": "change",
                        "kind": "workspace_change",
                        "description": "Change the file.",
                        "required": True,
                        "params": {"target_path": "app.py"},
                    },
                    {
                        "id": "run_after",
                        "kind": "local_execution",
                        "description": "Run after editing.",
                        "required": True,
                        "params": {"target_path": "app.py"},
                    },
                ],
                "constraints": [
                    {"kind": "before", "first": "run_before", "second": "change"},
                    {"kind": "before", "first": "change", "second": "run_after"},
                ],
            },
            fallback=fallback,
            task="change app.py, run it again, and display results",
        )

        executions = [obligation for obligation in contract.obligations if obligation.kind == "local_execution"]

        self.assertEqual([obligation.id for obligation in executions], ["run_after"])
        self.assertFalse(any(constraint.first == "run_before" for constraint in contract.constraints))

    def test_model_contract_drops_evidence_only_ordering_constraints(self):
        fallback = derive_task_contract(
            task="run hello.py and display the output",
            intent="shell",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="hello.py",
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "run_program",
                        "kind": "local_execution",
                        "description": "Run the program.",
                        "required": True,
                        "params": {"target_path": "hello.py"},
                    },
                    {
                        "id": "show_output",
                        "kind": "visible_output",
                        "description": "Display command output.",
                        "required": True,
                    },
                    {
                        "id": "respond",
                        "kind": "assistant_response",
                        "description": "Respond conversationally.",
                        "required": True,
                    },
                ],
                "constraints": [
                    {"kind": "before", "first": "run_program", "second": "show_output"},
                    {"kind": "before", "first": "show_output", "second": "respond"},
                ],
            },
            fallback=fallback,
            task="run hello.py and display the output, then tell me how you feel",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "hello\n", "stderr": ""},
            }
        ]

        self.assertEqual(contract.constraints, ())
        self.assertEqual(contract_missing(contract, observations), [])

    def test_model_contract_infers_target_path_from_python_command(self):
        fallback = derive_task_contract(
            task="run hello.py",
            intent="shell",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
        )
        contract = contract_from_model_json(
            {
                "obligations": [
                    {
                        "id": "run_program",
                        "kind": "local_execution",
                        "description": "Run the program.",
                        "required": True,
                        "params": {"command": "python hello.py"},
                    }
                ],
                "constraints": [],
            },
            fallback=fallback,
            task="run hello.py",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "hello\n", "stderr": ""},
            }
        ]

        self.assertEqual(contract_missing(contract, observations), [])


if __name__ == "__main__":
    unittest.main()
