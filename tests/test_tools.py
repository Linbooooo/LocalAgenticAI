import tempfile
import unittest
from pathlib import Path

from local_agent.config import AgentConfig
from local_agent.tools import WorkspaceTools


def make_tools(workspace: Path) -> WorkspaceTools:
    config = AgentConfig(workspace=workspace, trust="auto")
    config.finalize()
    return WorkspaceTools(config)


class ToolTests(unittest.TestCase):
    def test_file_tools_stay_in_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = make_tools(Path(directory))
            result = tools.write_file("hello.txt", "hello\nworld\n")
            self.assertTrue(result["ok"])

            read = tools.read_file("hello.txt")
            self.assertIn("1: hello", read["content"])

            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                tools.read_file("../outside.txt")

    def test_shell_blocks_network_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = make_tools(Path(directory))
            result = tools.run_shell("curl https://example.com")
            self.assertFalse(result["ok"])
            self.assertIn("local-only", result["error"])

    def test_shell_can_run_python_script(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hello_world.py").write_text("print('hello world')\n", encoding="utf-8")
            tools = make_tools(Path(directory))
            result = tools.run_shell("python hello_world.py", timeout_seconds=10)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["stdout"].strip(), "hello world")

    def test_search_text(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = make_tools(Path(directory))
            tools.write_file("src/app.py", "def hello():\n    return 'hi'\n")
            result = tools.search_text("hello", path=".", file_glob="*.py")
            self.assertTrue(result["ok"])
            self.assertEqual(result["matches"][0]["path"], "src/app.py")


if __name__ == "__main__":
    unittest.main()
