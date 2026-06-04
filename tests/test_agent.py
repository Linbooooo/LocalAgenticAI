import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_agent.agent import (
    CompletionCriteria,
    LocalAgent,
    _completion_missing,
    _normalize_written_content,
    _task_spec,
    _validate_action,
)
from local_agent.config import AgentConfig


def make_agent(workspace: Path) -> LocalAgent:
    config = AgentConfig(workspace=workspace, trust="auto")
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


class AgentTests(unittest.TestCase):
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
            self.assertEqual(result.turns, 3)

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
            self.assertEqual(result.turns, 3)

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
            self.assertEqual(result.turns, 3)

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
            self.assertEqual(result.turns, 3)

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

            self.assertIn("IndexError", result.content)
            self.assertIn("13", result.content)
            self.assertNotIn("Please provide an input value.", result.content)
            self.assertIn("$ python3 fibonacci.py 7", result.content)
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

            result = agent.run("update project files for the Python hello example")

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

    def test_run_the_file_uses_last_written_file(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello world')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello.py"

            result = agent.run("run the file for me")

            self.assertEqual(result.content.strip(), "hello world")

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
            self.assertEqual(agent.client.chat.call_count, 3)

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


if __name__ == "__main__":
    unittest.main()
