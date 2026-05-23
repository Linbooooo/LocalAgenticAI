import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from local_agent.agent import LocalAgent
from local_agent.config import AgentConfig


def make_agent(workspace: Path) -> LocalAgent:
    config = AgentConfig(workspace=workspace, trust="auto")
    config.finalize()
    return LocalAgent(config)


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
            result = agent.run("write a small Python hello file")

            self.assertEqual(result.content, "Created hello.py.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')
            self.assertEqual(agent.last_written_file, "hello.py")
            self.assertEqual(result.turns, 2)

    def test_action_loop_can_write_then_run(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                return_value={
                    "message": {
                        "role": "assistant",
                        "content": '{"action":"write_file","path":"hello.py","content":"print(\\"ok\\")\\n"}',
                    }
                }
            )

            result = agent.run("write a hello file and test it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("ok")\n')
            self.assertIn("Completed and verified successfully.", result.content)
            self.assertIn("$ python3 hello.py", result.content)
            self.assertIn("ok", result.content)
            self.assertEqual(agent.last_written_file, "hello.py")
            self.assertEqual(result.turns, 2)
            self.assertEqual(agent.client.chat.call_count, 1)

    def test_successful_run_stops_before_model_can_overedit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
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
            self.assertEqual(agent.client.chat.call_count, 1)

    def test_repeated_identical_write_does_not_loop_forever(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
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

            result = agent.run("write a small Python hello file")

            self.assertEqual(result.content, "Created hello.py.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')
            self.assertEqual(result.turns, 3)

    def test_invalid_prose_action_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
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
            self.assertEqual(agent.client.chat.call_count, 2)

    def test_missing_required_action_field_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                side_effect=[
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

            self.assertEqual(result.content, "Created hello.py.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("field ok")\n')
            self.assertEqual(agent.client.chat.call_count, 3)

    def test_python_code_block_is_salvaged_to_named_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = make_agent(Path(directory))
            agent.client.chat = Mock(
                return_value={
                    "message": {
                        "role": "assistant",
                        "content": 'Here is the file:\n```python\nprint("salvaged")\n```',
                    }
                }
            )

            result = agent.run("write a new file named hello.py run it")

            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("salvaged")\n')
            self.assertIn("salvaged", result.content)
            self.assertEqual(agent.client.chat.call_count, 1)

    def test_run_the_file_uses_last_written_file(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello world')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello.py"

            result = agent.run("run the file for me")

            self.assertEqual(result.content.strip(), "hello world")


if __name__ == "__main__":
    unittest.main()
