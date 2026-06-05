import unittest

from local_agent.task_contract import contract_missing, derive_task_contract, format_contract_evidence_for_final


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

        self.assertIn("Only 1 of 3 requested successful local executions have been observed.", missing)

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


if __name__ == "__main__":
    unittest.main()
