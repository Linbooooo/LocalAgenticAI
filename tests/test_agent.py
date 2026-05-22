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
                return_value={
                    "message": {
                        "role": "assistant",
                        "content": '{"action":"write_file","path":"hello.py","content":"print(\\"hi\\")\\n","message":"Created hello.py."}',
                    }
                }
            )
            result = agent.run("write a small Python hello file")

            self.assertEqual(result.content, "Created hello.py.")
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), 'print("hi")\n')
            self.assertEqual(agent.last_written_file, "hello.py")

    def test_run_the_file_uses_last_written_file(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello.py").write_text("print('hello world')\n", encoding="utf-8")
            agent = make_agent(Path(directory))
            agent.last_written_file = "hello.py"

            result = agent.run("run the file for me")

            self.assertEqual(result.content.strip(), "hello world")


if __name__ == "__main__":
    unittest.main()
