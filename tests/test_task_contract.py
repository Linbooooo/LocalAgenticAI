import unittest

from local_agent.task_contract import contract_from_model_json, contract_missing, derive_task_contract, format_contract_evidence_for_final


class TaskContractTests(unittest.TestCase):
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
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "old\n", "stderr": ""},
            },
            {
                "step": 2,
                "action": {"action": "replace_in_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py"},
            },
        ]
        complete = [
            *incomplete,
            {
                "step": 3,
                "action": {"action": "run_shell", "command": "python3 hello.py"},
                "result": {"ok": True, "returncode": 0, "stdout": "new\n", "stderr": ""},
            },
            {
                "step": 4,
                "action": {"action": "delete_file", "path": "hello.py"},
                "result": {"ok": True, "path": "hello.py"},
            },
        ]

        self.assertTrue(contract_missing(contract, incomplete))
        self.assertEqual(contract_missing(contract, complete), [])


if __name__ == "__main__":
    unittest.main()
