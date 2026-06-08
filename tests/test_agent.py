import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_agent.agent import (
    CompletionCriteria,
    LocalAgent,
    _command_signature,
    _completion_missing,
    _completion_criteria,
    _deterministic_edit_scope_issues,
    _format_repair_guidance,
    _normalize_written_content,
    _looks_like_edit_or_test_creation_request,
    _python_run_command_for_path,
    _repair_python_assert_expected_literals,
    _review_failure_is_external_evidence_only,
    _strip_trailing_question,
    _task_spec,
    _tool_intent_for_action,
    _validate_action,
    _validate_action_against_contract,
)
from local_agent.config import AgentConfig
from local_agent.task_contract import derive_task_contract


def make_agent(workspace: Path) -> LocalAgent:
    config = AgentConfig(workspace=workspace, trust="auto", contract_mode="fallback")
    config.finalize()
    return LocalAgent(config)


def make_model_contract_agent(workspace: Path) -> LocalAgent:
    config = AgentConfig(workspace=workspace, trust="auto", contract_mode="model")
    config.finalize()
    return LocalAgent(config)


def route_response(mode: str, *, requires_run: bool = False, confidence: float = 0.95):
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "mode": mode,
                    "requires_run": requires_run,
                    "confidence": confidence,
                    "reason": "test route",
                }
            ),
        }
    }


def contract_response(obligations: list[dict], constraints: list[dict] | None = None):
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps({"obligations": obligations, "constraints": constraints or []}),
        }
    }


def review_response(satisfied: bool, reason: str = "reviewed", missing: list[str] | None = None):
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps({"satisfied": satisfied, "reason": reason, "missing": missing or []}),
        }
    }


class AgentTests(unittest.TestCase):
    def test_strip_trailing_question_preserves_direct_answer(self):
        text = "I completed the run. I'm feeling focused today. How about yourself?"

        self.assertEqual(_strip_trailing_question(text), "I completed the run. I'm feeling focused today.")

    def test_direct_shell_command_bypasses_model(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            result = agent.run('execute "python3 -c \'print(123)\'"')
            self.assertEqual(result.content.strip(), "123")

    def test_edit_request_writes_file_from_model_json(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"hi\\")\\n"}',
                        }
                    },
                ]
            )
            result = agent.run("write a small Python hello file")

            self.assertEqual(result.content, "Created `hello.py`.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')
            self.assertEqual(agent.last_written_file, "hello.py")
            self.assertEqual(result.turns, 1)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_action_loop_can_write_then_run(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"ok\\")\\n"}',
                        }
                    },
                ]
            )

            result = agent.run("write a hello file and test it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("ok")\n')
            self.assertIn("Completed and verified successfully.", result.content)
            self.assertIn("$ python3 hello.py", result.content)
            self.assertIn("ok", result.content)
            self.assertEqual(agent.last_written_file, "hello.py")
            self.assertEqual(result.turns, 2)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_existing_file_update_reads_then_reviews_before_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "normalizer.py")
            path.write_text("def normalize_name(value):\n    return value.strip().lower()\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "replace_in_file",
                                    "path": "normalizer.py",
                                    "old": "return value.strip().lower()",
                                    "new": "return '-'.join(value.strip().lower().split())",
                                }
                            ),
                        }
                    },
                    review_response(True, "The function now lowercases and hyphenates whitespace."),
                ]
            )

            result = agent.run("modify normalizer.py so normalize_name lowercases and replaces spaces with hyphens")

            self.assertIn("Completed all requested local steps.", result.content)
            self.assertIn("'-'.join", path.read_text(encoding="utf-8"))
            self.assertEqual(result.turns, 3)
            self.assertEqual(agent.client.chat.call_count, 3)
            action_context = agent.client.chat.call_args_list[1].kwargs["messages"][1]["content"]
            self.assertIn("def normalize_name(value):", action_context)
            review_context = agent.client.chat.call_args_list[2].kwargs["messages"][1]["content"]
            self.assertIn("Original source before this edit:", review_context)
            self.assertIn("Edited source:", review_context)

    def test_failed_edit_review_requires_corrective_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "pricing.py")
            path.write_text("def total(price, tax):\n    return price + tax\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "replace_in_file",
                                    "path": "pricing.py",
                                    "old": "return price + tax",
                                    "new": "return price",
                                }
                            ),
                        }
                    },
                    review_response(False, "The edit removed tax instead of applying a percentage.", ["tax is ignored"]),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "path": "pricing.py",
                                    "content": "def total(price, tax):\n    return price * (1 + tax)\n",
                                }
                            ),
                        }
                    },
                    review_response(True, "The function now applies tax as a percentage multiplier."),
                ]
            )

            result = agent.run("change pricing.py so total treats tax as a percentage rate")

            self.assertIn("Completed all requested local steps.", result.content)
            self.assertIn("return price * (1 + tax)", path.read_text(encoding="utf-8"))
            self.assertEqual(result.turns, 6)
            self.assertEqual(agent.client.chat.call_count, 5)
            corrective_context = agent.client.chat.call_args_list[3].kwargs["messages"][1]["content"]
            self.assertIn("Repair a code edit that failed semantic review", corrective_context)
            self.assertIn("tax is ignored", corrective_context)

    def test_test_only_edit_preserves_implementation_and_reports_final_evidence_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "two_sum.py")
            original = (
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for index, value in enumerate(nums):\n"
                "        if target - value in seen:\n"
                "            return [seen[target - value], index]\n"
                "        seen[value] = index\n"
                "    return []\n\n"
                "if __name__ == '__main__':\n"
                "    print(two_sum([2, 7, 11, 15], 9))\n"
            )
            wrong = (
                "def new_two_sum(nums, target):\n"
                "    for left in range(len(nums)):\n"
                "        for right in range(left + 1, len(nums)):\n"
                "            if nums[left] + nums[right] == target:\n"
                "                return [left, right]\n"
                "    return []\n\n"
                "if __name__ == '__main__':\n"
                "    print(new_two_sum([1, 4, 6, 8], 10))\n"
            )
            corrected = original.replace(
                "    print(two_sum([2, 7, 11, 15], 9))",
                "    cases = [([1, 4, 6, 8], 10), ([3, 3, 9], 6), ([-4, 1, 5], 1)]\n"
                "    for nums, target in cases:\n"
                "        result = two_sum(nums, target)\n"
                "        print(nums, target, result)",
            )
            path.write_text(original, encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"action": "write_file", "path": "two_sum.py", "content": wrong}),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"path": "two_sum.py", "content": corrected}),
                        }
                    },
                    review_response(True, "Only the test/demo cases changed; two_sum is preserved."),
                ]
            )

            result = agent.run(
                "go into two_sum.py and change the test to a completely different, more complicated one. "
                "Run it and display the code and results."
            )

            self.assertEqual(path.read_text(encoding="utf-8"), corrected)
            self.assertIn("def two_sum(nums, target):", result.content)
            self.assertNotIn("def new_two_sum", result.content)
            self.assertEqual(result.content.count("`two_sum.py`:"), 1)
            self.assertEqual(result.content.count("$ python3 two_sum.py"), 1)
            self.assertIn("[-4, 1, 5] 1 [0, 2]", result.content)
            self.assertEqual(result.turns, 8)

    def test_test_only_scope_rejects_implementation_rewrite(self):
        original = "def two_sum(nums, target):\n    return []\n\nprint(two_sum([2, 7], 9))\n"
        edited = "def new_two_sum(nums, target):\n    return [0, 1]\n\nprint(new_two_sum([2, 7], 9))\n"

        issues = _deterministic_edit_scope_issues("two_sum.py", "tests_only", original, edited)

        self.assertTrue(any("two_sum" in issue for issue in issues))

    def test_edit_review_defers_runtime_only_complaints_to_evidence_ledger(self):
        self.assertTrue(
            _review_failure_is_external_evidence_only(
                "The source looks correct, but runtime proof is missing.",
                ["test execution and output"],
            )
        )
        self.assertTrue(
            _review_failure_is_external_evidence_only(
                "The edit is complete, but reporting remains.",
                ["test execution", "final code display", "results display"],
            )
        )
        self.assertFalse(
            _review_failure_is_external_evidence_only(
                "The source changed the implementation.",
                ["two_sum was renamed", "tests were not run"],
            )
        )

    def test_uninvoked_free_test_function_gets_explicit_standard_library_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.py")
            path.write_text(
                "def test_sample():\n"
                "    assert 2 + 2 == 4\n\n"
                "if __name__ == '__main__':\n"
                "    print('demo')\n",
                encoding="utf-8",
            )

            command = _python_run_command_for_path("sample.py", path)

            self.assertIn("runpy.run_path", command)
            self.assertTrue(command.endswith(" sample.py"))

    def test_test_function_called_from_main_keeps_direct_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.py")
            path.write_text(
                "def test_sample():\n"
                "    assert 2 + 2 == 4\n\n"
                "if __name__ == '__main__':\n"
                "    test_sample()\n"
                "    print('All tests passed.')\n",
                encoding="utf-8",
            )

            self.assertEqual(_python_run_command_for_path("sample.py", path), "python3 sample.py")

    def test_unittest_case_in_regular_module_uses_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.py")
            path.write_text(
                "import unittest\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _python_run_command_for_path("sample.py", path),
                "python3 -m unittest discover -s . -p sample.py",
            )

    def test_test_only_assertion_repair_uses_preserved_implementation_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "two_sum.py")
            path.write_text(
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for index, value in enumerate(nums):\n"
                "        if target - value in seen:\n"
                "            return [seen[target - value], index]\n"
                "        seen[value] = index\n"
                "    return []\n\n"
                "def test_two_sum():\n"
                "    assert two_sum([1, 2, 3, 4, 5], 10) == [3, 4]\n",
                encoding="utf-8",
            )

            repaired = _repair_python_assert_expected_literals(path)

            self.assertIsNotNone(repaired)
            self.assertIn("def two_sum(nums, target):", repaired)
            self.assertIn("two_sum([1, 2, 3, 4, 5], 10) == []", repaired)

    def test_written_unittest_file_runs_with_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"action":"write_file","path":"tests/test_sample.py",'
                                '"content":"import unittest\\n\\nclass SampleTests(unittest.TestCase):\\n'
                                '    def test_ok(self):\\n        self.assertEqual(1 + 1, 2)\\n\\n'
                                'if __name__ == \\"__main__\\":\\n    unittest.main()\\n"}'
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write tests/test_sample.py and run it")

            self.assertIn("$ python3 -m unittest discover -s tests -p test_sample.py", result.content)
            self.assertIn("OK", result.content)
            self.assertEqual(result.turns, 2)

    def test_inline_assert_tests_can_satisfy_test_request(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            content = (
                "def merge_sort(values):\n"
                "    return sorted(values)\n\n"
                "def test_merge_sort():\n"
                "    assert merge_sort([3, 1, 2]) == [1, 2, 3]\n"
                "    assert merge_sort([]) == []\n\n"
                "if __name__ == '__main__':\n"
                "    test_merge_sort()\n"
                "    print('All tests passed.')\n"
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "merge_sort.py",
                                    "content": content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python file that implements merge sort and include test cases. Run it and show results.")

            self.assertIn("$ python3 merge_sort.py", result.content)
            self.assertIn("All tests passed.", result.content)
            self.assertIn("Completed and verified successfully.", result.content)
            self.assertEqual(result.turns, 2)

    def test_empty_successful_run_does_not_satisfy_requested_results(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            initial_content = "def three_sum(nums):\n    return []\n"
            final_content = "def three_sum(nums):\n    return [[-1, 0, 1]]\n\nprint(three_sum([-1, 0, 1]))\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": initial_content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 three_sum.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": final_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python code that implements 3-sum and test it and display the results")

            self.assertEqual(Path(directory, "three_sum.py").read_text(encoding="utf-8"), final_content)
            self.assertIn("$ python3 three_sum.py", result.content)
            self.assertIn("[[-1, 0, 1]]", result.content)
            self.assertEqual(result.content.count("$ python3 three_sum.py"), 1)
            self.assertEqual(result.turns, 4)

    def test_failed_unittest_output_does_not_satisfy_test_completion(self):
        observations = [
            {
                "step": 1,
                "action": {"action": "write_file", "path": "example.py"},
                "result": {"ok": True, "path": "example.py"},
            },
            {
                "step": 2,
                "action": {"action": "run_shell", "command": "python3 example.py"},
                "result": {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "E\n======================================================================\nERROR: test_bad\n\nFAILED (errors=1)",
                },
            },
        ]

        missing = _completion_missing(
            CompletionCriteria(requires_run=True, requires_tests=True, requires_output=True),
            observations,
        )

        self.assertIn("Tests were requested, but no passing test command/output evidence was observed.", missing)

    def test_passing_unittest_output_satisfies_test_completion(self):
        observations = [
            {
                "step": 1,
                "action": {"action": "write_file", "path": "example.py"},
                "result": {"ok": True, "path": "example.py"},
            },
            {
                "step": 2,
                "action": {"action": "run_shell", "command": "python3 example.py"},
                "result": {
                    "ok": True,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "...\n----------------------------------------------------------------------\nRan 3 tests in 0.000s\n\nOK",
                },
            },
        ]

        missing = _completion_missing(
            CompletionCriteria(requires_run=True, requires_tests=True, requires_output=True),
            observations,
        )

        self.assertEqual(missing, [])

    def test_none_results_phrase_does_not_require_command_output(self):
        criteria = _completion_criteria(
            "modify cache.py so get_or_create does not store None results in the cache",
            route_requires_run=True,
            intent="edit",
        )

        self.assertFalse(criteria.requires_run)
        self.assertFalse(criteria.requires_output)

    def test_explicit_display_results_requires_output(self):
        criteria = _completion_criteria(
            "modify cache.py, run it, and display the results",
            route_requires_run=True,
            intent="edit",
        )

        self.assertTrue(criteria.requires_run)
        self.assertTrue(criteria.requires_output)

    def test_shell_route_can_require_run_without_execution_keyword(self):
        criteria = _completion_criteria("benchmark the local model", route_requires_run=True, intent="shell")

        self.assertTrue(criteria.requires_run)

    def test_silent_run_is_rejected_before_execution_when_output_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            initial_content = "def three_sum(nums):\n    return []\n"
            final_content = "def three_sum(nums):\n    return []\n\nprint(three_sum([]))\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": initial_content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 three_sum.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 three_sum.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": final_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python code that implements 3-sum and display the results")

            self.assertIn("[]", result.content)
            self.assertEqual(result.content.count("$ python3 three_sum.py"), 1)
            self.assertEqual(result.turns, 4)

    def test_repeated_failed_run_is_rejected_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            content = "print('starting')\nraise RuntimeError('boom')\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "broken.py",
                                    "content": content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 broken.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 broken.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"No fix."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"No fix."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"No fix."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"No fix."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"No fix."}',
                        }
                    },
                ]
            )

            result = agent.run("write broken.py run it and display the result")

            self.assertEqual(result.content.count("RuntimeError: boom"), 1)
            self.assertIn("Step 3: read_file completed", result.content)
            self.assertIn("Could not repair the failed verification", result.content)

    def test_repair_guidance_mentions_algorithm_test_oracles(self):
        observations = [
            {
                "step": 1,
                "action": {"action": "write_file", "path": "two_sum.py"},
                "result": {"ok": True, "path": "two_sum.py"},
            },
            {
                "step": 2,
                "action": {"action": "run_shell", "command": "python3 two_sum.py"},
                "result": {
                    "ok": False,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "AssertionError: expected [3, 4], got []",
                },
            },
        ]

        guidance = _format_repair_guidance(CompletionCriteria(requires_run=True), observations)

        self.assertIn("brute-force oracle", guidance)
        self.assertIn("returned-index problems", guidance)

    def test_python_command_signature_normalizes_equivalent_script_runs(self):
        self.assertEqual(_command_signature("python3 ./two_sum.py"), _command_signature("python3 -u two_sum.py"))
        self.assertEqual(_command_signature("python two_sum.py"), "python two_sum.py")

    def test_finish_is_rejected_when_completion_evidence_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            initial_content = "def three_sum(nums):\n    return []\n"
            final_content = "def three_sum(nums):\n    return []\n\nprint(three_sum([]))\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": initial_content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"finish","message":"Done."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": final_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python code that implements 3-sum and display the results")

            self.assertIn("[]", result.content)
            self.assertNotEqual(result.content, "Done.")
            self.assertIn("Completed and verified successfully.", result.content)
            self.assertEqual(result.turns, 4)

    def test_answer_is_rejected_after_progress_when_completion_evidence_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            initial_content = "def three_sum(nums):\n    return []\n"
            final_content = "def three_sum(nums):\n    return []\n\nprint(three_sum([]))\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": initial_content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"Please provide the existing code."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": final_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python code that implements 3-sum and display the results")

            self.assertIn("[]", result.content)
            self.assertNotIn("Please provide the existing code.", result.content)
            self.assertIn("Completed and verified successfully.", result.content)
            self.assertEqual(result.turns, 4)

    def test_failed_run_requires_repair_action_before_answering(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            content = (
                "import sys\n\n"
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "if __name__ == '__main__':\n"
                "    n = int(sys.argv[1])\n"
                "    print(fibonacci(n))\n"
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "fibonacci.py",
                                    "content": content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"answer","message":"Please provide an input value."}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 fibonacci.py 7","timeout_seconds":120}',
                        }
                    },
                ]
            )

            result = agent.run("write a python file that outputs the nth fibonacci number given an input n. test it and display the results")

            self.assertNotIn("IndexError", result.content)
            self.assertIn("13", result.content)
            self.assertNotIn("Please provide an input value.", result.content)
            self.assertIn("$ python3 fibonacci.py 7", result.content)
            self.assertEqual(result.content.count("$ python3"), 1)
            self.assertEqual(result.turns, 4)

    def test_input_script_can_be_verified_with_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            content = (
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "if __name__ == '__main__':\n"
                "    n = int(input('Enter the value of n: '))\n"
                "    print(fibonacci(n))\n"
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "fibonacci.py",
                                    "content": content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "run_shell",
                                    "command": "python3 fibonacci.py",
                                    "stdin": "6\n",
                                    "timeout_seconds": 120,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python file that outputs the nth fibonacci number given an input n. test it and display the results")

            self.assertIn("8", result.content)
            self.assertNotIn("EOFError", result.content)
            self.assertIn("$ python3 fibonacci.py", result.content)
            self.assertEqual(result.turns, 2)

    def test_failed_run_recovers_by_reading_file_after_invalid_repair_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            initial_content = (
                "import sys\n\n"
                "def fibonacci(n):\n"
                "    return n\n\n"
                "if __name__ == '__main__':\n"
                "    n = int(sys.argv[1])\n"
                "    print(fibonacci(n))\n"
            )
            fixed_content = (
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "if __name__ == '__main__':\n"
                "    print(fibonacci(6))\n"
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "fibonacci.py",
                                    "content": initial_content,
                                }
                            ),
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "fibonacci.py",
                                    "content": fixed_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python file that outputs the nth fibonacci number given an input n. test it and display the results")

            self.assertEqual(Path(directory, "fibonacci.py").read_text(encoding="utf-8"), fixed_content)
            self.assertIn("8", result.content)
            self.assertNotIn("Need input.", result.content)
            self.assertEqual(result.turns, 5)

    def test_failed_run_without_corrective_action_stops_before_max_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            content = "print('starting')\nraise RuntimeError('boom')\n"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "broken.py",
                                    "content": content,
                                }
                            ),
                        }
                    },
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I still cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I still cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"I still cannot fix it."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"No fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"No fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"No fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                    {"message": {"role": "assistant", "content": '{"action":"answer","message":"Still no fix."}'}},
                ]
            )

            result = agent.run("write broken.py run it and display the result")

            self.assertIn("Could not repair the failed verification", result.content)
            self.assertIn("RuntimeError: boom", result.content)
            self.assertLess(result.turns, agent.config.max_steps)

    def test_missing_unittest_start_dir_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            action = {
                "action": "run_shell",
                "command": "python3 -m unittest discover -s tests -p test_valid_parentheses.py",
                "timeout_seconds": 120,
            }

            error = _validate_action(
                action,
                "edit",
                CompletionCriteria(requires_run=True, requires_output=True),
                [],
                Path(directory),
            )

            self.assertIsNotNone(error)
            self.assertIn("start directory does not exist", error)

    def test_failed_bad_test_command_recovers_by_running_latest_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            Path(directory, "valid_parentheses.py").write_text("print('all checks passed')\n", encoding="utf-8")
            agent.last_written_file = "valid_parentheses.py"
            observations = [
                {
                    "step": 1,
                    "action": {"action": "write_file", "path": "valid_parentheses.py"},
                    "result": {"ok": True, "path": "valid_parentheses.py"},
                },
                {
                    "step": 2,
                    "action": {
                        "action": "run_shell",
                        "command": "python3 -m unittest discover -s tests -p test_valid_parentheses.py",
                    },
                    "result": {"ok": False, "returncode": 1, "stderr": "ImportError: tests"},
                },
            ]

            recovery = agent._repair_recovery_action(
                CompletionCriteria(requires_run=True, requires_output=True),
                observations,
            )

            self.assertEqual(
                recovery,
                {"action": "run_shell", "command": "python3 valid_parentheses.py", "timeout_seconds": 120},
            )

    def test_successful_run_stops_before_model_can_overedit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"stable\\")\\n"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"raise Exception(\\"broken\\")\\n"}',
                        }
                    },
                ]
            )

            result = agent.run("write a hello file and test it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("stable")\n')
            self.assertIn("stable", result.content)
            self.assertEqual(result.turns, 2)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_repeated_identical_write_does_not_loop_forever(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"hi\\")\\n"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"hi\\")\\n"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"finish","message":"Created hello.py."}',
                        }
                    },
                ]
            )

            result = agent.run("write project files for the Python hello example")

            self.assertEqual(result.content, "Created hello.py.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')
            self.assertEqual(result.turns, 3)

    def test_invalid_prose_action_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sure, I can do that. I will create the file now.",
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"retry ok\\")\\n"}',
                        }
                    },
                ]
            )

            result = agent.run("write a new file named hello.py run it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("retry ok")\n')
            self.assertIn("retry ok", result.content)
            self.assertEqual(agent.client.chat.call_count, 3)

    def test_missing_required_action_field_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"field ok\\")\\n"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"finish","message":"Created hello.py."}',
                        }
                    },
                ]
            )

            result = agent.run("write a new file named hello.py")

            self.assertEqual(result.content, "Created `hello.py`.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("field ok")\n')
            self.assertEqual(agent.client.chat.call_count, 3)

    def test_python_code_block_is_salvaged_to_named_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'Here is the file:\n```python\nprint("salvaged")\n```',
                        }
                    },
                ]
            )

            result = agent.run("write a new file named hello.py run it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("salvaged")\n')
            self.assertIn("salvaged", result.content)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_python_code_block_is_salvaged_to_inferred_algorithm_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": "```python\ndef three_sum(nums):\n    return []\n```",
                        }
                    },
                ]
            )

            result = agent.run("write a python file that solves 3sum")

            self.assertEqual(Path(directory, "three_sum.py").read_text(encoding="utf-8"), "def three_sum(nums):\n    return []\n")
            self.assertEqual(result.content, "Created `three_sum.py`.")
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_python_code_block_is_salvaged_to_planned_generic_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'Here is the file:\n```python\nprint("Hello World")\n```',
                        }
                    },
                ]
            )

            result = agent.run("write a python file that prints out Hello World")

            self.assertEqual(Path(directory, "hello_world.py").read_text(encoding="utf-8"), 'print("Hello World")\n')
            self.assertEqual(result.content, "Created `hello_world.py`.")
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_declined_write_does_not_poison_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            task = "write a python file that prints hello world"
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello_world.py","content":"print(\\"hello world\\")\\n"}',
                        }
                    },
                    route_response("edit"),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello_world.py","content":"print(\\"hello world\\")\\n"}',
                        }
                    },
                ]
            )
            agent.tools.write_file = Mock(
                side_effect=[
                    {"ok": False, "error": "User declined tool execution."},
                    {"ok": True, "path": "hello_world.py"},
                ]
            )

            declined = agent.run(task)
            retried = agent.run(task)

            self.assertIn("User declined tool execution.", declined.content)
            self.assertEqual(retried.content, "Created `hello_world.py`.")
            self.assertFalse(
                any("User declined tool execution." in str(message.get("content", "")) for message in agent.messages)
            )

    def test_action_protocol_does_not_receive_stale_chat_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.messages.append({"role": "user", "content": "test hello_world.py"})
            agent.messages.append(
                {
                    "role": "assistant",
                    "content": "Completed and verified successfully.\n\nCommand results:\n$ python3 hello_world.py\nHello, World!",
                }
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"two_sum.py","content":"print(\\"ok\\")\\n"}',
                        }
                    },
                ]
            )

            result = agent.run("write a code that solve the 2sum problem and test it and display the results.")

            self.assertIn("ok", result.content)
            route_messages = agent.client.chat.call_args_list[0].kwargs["messages"]
            action_messages = agent.client.chat.call_args_list[1].kwargs["messages"]
            self.assertEqual(len(route_messages), 2)
            self.assertEqual(len(action_messages), 2)
            self.assertNotIn("Completed and verified successfully", route_messages[1]["content"])
            self.assertNotIn("Completed and verified successfully", action_messages[1]["content"])
            self.assertIn("write a code that solve the 2sum problem", action_messages[1]["content"])

    def test_written_content_strips_outer_code_fence(self):
        content = "```python\nprint('ok')\n```"

        self.assertEqual(_normalize_written_content(content), "print('ok')\n")

    def test_written_content_decodes_literal_newline_payload(self):
        content = "\\ndef add(a, b):\\n\\treturn a + b\\n"

        self.assertEqual(_normalize_written_content(content), "\ndef add(a, b):\n\treturn a + b\n")

    def test_model_router_keeps_greeting_in_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("chat"),
                    {"message": {"role": "assistant", "content": "Hello!"}},
                ]
            )

            result = agent.run("hello there")

            self.assertEqual(result.content, "Hello!")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_low_confidence_action_route_downgrades_to_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", confidence=0.2),
                    {"message": {"role": "assistant", "content": "Could you clarify what you want changed?"}},
                ]
            )

            result = agent.run("maybe something")

            self.assertEqual(result.content, "Could you clarify what you want changed?")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_run_the_file_uses_state_aware_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello world')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello.py"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("shell", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "run_shell",
                                    "command": "python3 hello.py",
                                    "timeout_seconds": 120,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("run the file for me")

            self.assertIn("Completed and verified successfully.", result.content)
            self.assertIn("hello world", result.content)
            action_context = agent.client.chat.call_args_list[1].kwargs["messages"][1]["content"]
            self.assertIn("User request:\nrun the file for me", action_context)
            self.assertIn('"last_written_file": "hello.py"', action_context)

    def test_test_it_uses_state_aware_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello_world.py").write_text("print('Hello, World!')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello_world.py"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("shell", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "run_shell",
                                    "command": "python3 hello_world.py",
                                    "stdin": "",
                                    "timeout_seconds": 120,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("test it.")

            self.assertIn("Completed and verified successfully.", result.content)
            self.assertIn("Hello, World!", result.content)
            action_context = agent.client.chat.call_args_list[1].kwargs["messages"][1]["content"]
            self.assertIn("User request:\ntest it.", action_context)
            self.assertIn('"last_written_file": "hello_world.py"', action_context)

    def test_shell_action_stops_after_successful_run(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello_world.py").write_text("print('Hello, World!')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("shell", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "run_shell",
                                    "command": "python3 hello_world.py",
                                    "stdin": "",
                                    "timeout_seconds": 120,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("test hello_world.py")

            self.assertIn("Completed and verified successfully.", result.content)
            self.assertIn("Hello, World!", result.content)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_write_tests_and_run_it_does_not_bypass_action_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory, "local_agent_demo")
            source_dir.mkdir()
            Path(source_dir, "two_sum.py").write_text(
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for index, value in enumerate(nums):\n"
                "        if target - value in seen:\n"
                "            return [seen[target - value], index]\n"
                "        seen[value] = index\n"
                "    return []\n",
                encoding="utf-8",
            )
            agent = make_agent(Path(directory))
            agent.last_written_file = "local_agent_demo/two_sum.py"
            test_content = (
                "import unittest\n\n"
                "from local_agent_demo.two_sum import two_sum\n\n"
                "class TwoSumTests(unittest.TestCase):\n"
                "    def test_pair(self):\n"
                "        self.assertEqual(two_sum([2, 7, 11, 15], 9), [0, 1])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "tests/test_two_sum.py",
                                    "content": test_content,
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("now write test cases and run it")

            self.assertTrue(Path(directory, "tests/test_two_sum.py").exists())
            self.assertIn("$ python3 -m unittest discover -s tests -p test_two_sum.py", result.content)
            self.assertNotIn("$ python3 local_agent_demo/two_sum.py", result.content)
            self.assertIn("OK", result.content)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_show_code_reads_last_written_file_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello world')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello.py"
            agent.client.chat = Mock()

            result = agent.run("show the code")

            self.assertIn("`hello.py`:", result.content)
            self.assertIn("print('hello world')", result.content)
            agent.client.chat.assert_not_called()

    def test_display_named_code_matches_hyphenated_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory, "local_agent_demo")
            source_dir.mkdir()
            Path(source_dir, "two_sum.py").write_text("def two_sum(nums, target):\n    return []\n", encoding="utf-8")
            test_dir = Path(directory, "tests")
            test_dir.mkdir()
            Path(test_dir, "test_two_sum.py").write_text("import unittest\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "tests/test_two_sum.py"
            agent.client.chat = Mock()

            result = agent.run("display the two-sum code")

            self.assertIn("`local_agent_demo/two_sum.py`:", result.content)
            self.assertIn("def two_sum", result.content)
            agent.client.chat.assert_not_called()

    def test_display_results_edit_request_does_not_read_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory, "local_agent/solutions")
            source_dir.mkdir(parents=True)
            Path(source_dir, "palindrome.py").write_text("print('old palindrome')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "local_agent/solutions/palindrome.py"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "write_file",
                                    "path": "three_sum.py",
                                    "content": "print([[-1, -1, 2], [-1, 0, 1]])\n",
                                }
                            ),
                        }
                    },
                ]
            )

            result = agent.run("write a python file that solves 3sum and test it. Display the results.")

            self.assertIn("[[-1, -1, 2], [-1, 0, 1]]", result.content)
            self.assertNotIn("old palindrome", result.content)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_display_command_results_does_not_trigger_source_read_shortcut(self):
        self.assertTrue(
            _looks_like_edit_or_test_creation_request(
                "run ordered_probe.py, change it, run it again, and display both command results"
            )
        )

    def test_contract_blocks_answer_after_target_read_while_edit_is_missing(self):
        contract = derive_task_contract(
            task="change app.py",
            intent="edit",
            requires_run=False,
            requires_tests=False,
            requires_output=False,
            target_path="app.py",
            operation="update",
        )
        observations = [
            {
                "step": 1,
                "action": {"action": "read_file", "path": "app.py"},
                "result": {"ok": True, "path": "app.py", "content": "1: print('old')"},
            }
        ]

        error = _validate_action_against_contract(
            {"action": "answer", "message": "Done."},
            contract,
            observations,
        )

        self.assertIn("requires a workspace change", error)

    def test_contract_authorizes_required_edit_in_mixed_shell_workflow(self):
        contract = derive_task_contract(
            task="run app.py, change it, and run it again",
            intent="edit",
            requires_run=True,
            requires_tests=False,
            requires_output=True,
            target_path="app.py",
            operation="update",
        )

        effective_intent = _tool_intent_for_action(
            "shell",
            {"action": "replace_in_file", "path": "app.py"},
            contract,
        )

        self.assertEqual(effective_intent, "edit")

    def test_fix_display_request_does_not_use_read_shortcut(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "math_bug.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "replace_in_file",
                                    "path": "math_bug.py",
                                    "old": "return a - b",
                                    "new": "return a + b",
                                }
                            ),
                        }
                    },
                    review_response(True, "The subtraction bug was replaced with addition."),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 -c \\"from math_bug import add; print(add(2, 3))\\"","timeout_seconds":120}',
                        }
                    },
                ]
            )

            result = agent.run("fix the bug in math_bug.py run it and display the result")

            self.assertIn("5", result.content)
            self.assertNotIn("return a - b", Path(directory, "math_bug.py").read_text(encoding="utf-8"))
            self.assertEqual(agent.client.chat.call_count, 4)

    def test_action_context_includes_relevant_coding_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            test_dir = Path(directory, "tests")
            test_dir.mkdir()
            Path(directory, "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
            Path(test_dir, "test_sample.py").write_text("import unittest\n", encoding="utf-8")
            agent = make_agent(Path(directory))

            context = agent._action_context(
                "run tests/test_sample.py",
                "shell",
                CompletionCriteria(requires_run=True, requires_output=True),
                [],
            )

            self.assertIn("Active coding skills:", context)
            self.assertIn("python-testing", context)
            self.assertIn("python3 -m unittest discover", context)

    def test_action_context_includes_planned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            spec = _task_spec("write a python file that prints out Hello World", "edit")

            context = agent._action_context(
                "write a python file that prints out Hello World",
                "edit",
                CompletionCriteria(),
                [],
                task_spec=spec,
            )

            self.assertEqual(spec.target_path, "hello_world.py")
            self.assertIn("Planned artifact:", context)
            self.assertIn("target_path: hello_world.py", context)
            self.assertIn("language: python", context)

    def test_task_contract_requires_requested_number_of_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("shell", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 hello.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 hello.py","timeout_seconds":120}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 hello.py","timeout_seconds":120}',
                        }
                    },
                ]
            )

            result = agent.run("run hello.py 3 times")

            self.assertEqual(result.content.count("$ python3 hello.py"), 3)
            self.assertEqual(result.turns, 3)
            self.assertEqual(agent.client.chat.call_count, 4)

    def test_task_contract_reports_read_source_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello from file')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("shell", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"list_files","path":".","max_depth":2,"limit":20}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"read_file","path":"hello.py","start_line":1,"max_lines":200}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"run_shell","command":"python3 hello.py","timeout_seconds":120}',
                        }
                    },
                ]
            )

            result = agent.run("scan this directory for python code. If you find it, display the code, and then run the code.")

            self.assertIn("Observed file contents:", result.content)
            self.assertIn("1: print('hello from file')", result.content)
            self.assertIn("$ python3 hello.py", result.content)
            self.assertIn("hello from file", result.content)
            self.assertEqual(result.turns, 3)

    def test_task_contract_forces_source_read_before_running_new_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"show_then_run.py","content":"print(\\"source-visible\\")\\n"}',
                        }
                    },
                ]
            )

            result = agent.run("write show_then_run.py that prints source-visible. Display the code, then run it.")

            self.assertIn("Observed file contents:", result.content)
            self.assertIn("1: print", result.content)
            self.assertIn("source-visible", result.content)
            self.assertEqual(result.content.count("$ python3 show_then_run.py"), 1)
            self.assertEqual(result.turns, 3)
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_edit_contract_rejects_finish_after_only_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "app.py").write_text("print('old')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit"),
                    {"message": {"role": "assistant", "content": '{"action":"finish","message":"Done."}'}},
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"replace_in_file","path":"app.py","old":"old","new":"new","max_replacements":1}',
                        }
                    },
                    review_response(True, "The requested update changed old to new."),
                ]
            )

            result = agent.run("update app.py")

            self.assertEqual(Path(directory, "app.py").read_text(encoding="utf-8"), "print('new')\n")
            self.assertNotEqual(result.content, "Done.")
            self.assertIn("Completed all requested local steps.", result.content)
            self.assertEqual(result.turns, 3)

    def test_model_contract_preserves_ordered_followup_on_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "hello_world.py")
            path.write_text("print('Hello, world!')\n", encoding="utf-8")
            agent = make_model_contract_agent(Path(directory))
            agent.last_written_file = "hello_world.py"
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    contract_response(
                        [
                            {
                                "id": "run_before_change",
                                "kind": "local_execution",
                                "description": "Run the existing program before changing it.",
                                "required": True,
                                "params": {"target_path": "hello_world.py", "min_successes": 1},
                                "evidence": ["successful run_shell"],
                            },
                            {
                                "id": "change_printout",
                                "kind": "workspace_change",
                                "description": "Change the printout to hello USA in the existing program.",
                                "required": True,
                                "params": {"target_path": "hello_world.py", "expected_text": "hello USA"},
                                "evidence": ["successful replace_in_file"],
                            },
                            {
                                "id": "run_after_change",
                                "kind": "local_execution",
                                "description": "Run the changed program.",
                                "required": True,
                                "params": {"target_path": "hello_world.py", "min_successes": 1},
                                "evidence": ["successful run_shell"],
                            },
                            {
                                "id": "delete_program",
                                "kind": "workspace_delete",
                                "description": "Delete the program file.",
                                "required": True,
                                "params": {"target_path": "hello_world.py"},
                                "evidence": ["successful delete_file"],
                            },
                        ],
                        [
                            {
                                "kind": "before",
                                "first": "run_before_change",
                                "second": "change_printout",
                                "description": "Run the original program before editing it.",
                            },
                            {
                                "kind": "before",
                                "first": "change_printout",
                                "second": "run_after_change",
                                "description": "Edit the program before running it again.",
                            },
                            {
                                "kind": "before",
                                "first": "run_after_change",
                                "second": "delete_program",
                                "description": "Run the changed program before deleting it.",
                            },
                        ],
                    ),
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"action":"replace_in_file","path":"hello_world.py",'
                                '"old":"Hello, world!","new":"hello USA","max_replacements":1}'
                            ),
                        }
                    },
                    review_response(True, "The printout was changed to hello USA."),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"delete_file","path":"hello_world.py"}',
                        }
                    },
                    {"message": {"role": "assistant", "content": '{"action":"finish","message":"Done."}'}},
                ]
            )

            result = agent.run(
                "run the program again. and then change the printout to hello USA. "
                "and then run it again. and then delete the file."
            )

            self.assertFalse(path.exists())
            self.assertIn("Hello, world!", result.content)
            self.assertIn("hello USA", result.content)
            self.assertEqual(result.content.count("$ python3 hello_world.py"), 2)
            self.assertEqual(result.turns, 6)

    def test_model_contract_assistant_response_prevents_auto_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_model_contract_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
                    route_response("edit", requires_run=True),
                    contract_response(
                        [
                            {
                                "id": "create_program",
                                "kind": "workspace_change",
                                "description": "Create a hello world program.",
                                "required": True,
                                "params": {"target_path": "hello.py"},
                            },
                            {
                                "id": "run_program",
                                "kind": "local_execution",
                                "description": "Run the hello world program.",
                                "required": True,
                                "params": {"target_path": "hello.py"},
                            },
                            {
                                "id": "answer_feeling",
                                "kind": "assistant_response",
                                "description": "Tell the user how you are feeling today.",
                                "required": True,
                                "params": {"expected_text": "feeling"},
                            },
                        ],
                        [
                            {"kind": "before", "first": "create_program", "second": "run_program"},
                            {"kind": "before", "first": "run_program", "second": "answer_feeling"},
                        ],
                    ),
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action":"write_file","path":"hello.py","content":"print(\\"Hello, world!\\")\\n"}',
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The program ran successfully. I am feeling focused today.",
                        }
                    },
                ]
            )

            result = agent.run("write and run a simple helloworld python program, and then tell me how you are feeling today.")

            self.assertIn("I am feeling focused today.", result.content)
            self.assertIn("$ python3 hello.py", result.content)
            self.assertEqual(result.turns, 2)


if __name__ == "__main__":
    unittest.main()
