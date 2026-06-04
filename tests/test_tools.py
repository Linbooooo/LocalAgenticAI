import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent.config import AgentConfig
from local_agent.tools import WorkspaceTools


def make_tools(workspace: Path) -> WorkspaceTools:
    config = AgentConfig(workspace=workspace, trust="auto")
    config.finalize()
    return WorkspaceTools(config)


def make_prompting_tools(workspace: Path) -> WorkspaceTools:
    config = AgentConfig(workspace=workspace, trust="ask")
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

    def test_shell_can_pass_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "echo_input.py").write_text("value = input()\nprint(f'got {value}')\n", encoding="utf-8")
            tools = make_tools(Path(directory))
            result = tools.run_shell("python3 echo_input.py", timeout_seconds=10, stdin="hello\n")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["stdout"].strip(), "got hello")

    def test_confirm_accepts_repeated_y_typo(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = make_prompting_tools(Path(directory))

            with patch("builtins.input", return_value="yy"):
                result = tools.write_file("hello.py", "print('hello')\n")

            self.assertTrue(result["ok"], result)
            self.assertEqual(Path(directory, "hello.py").read_text(encoding="utf-8"), "print('hello')\n")

    def test_shell_runs_test_files_with_unittest_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            test_dir = Path(directory, "tests")
            test_dir.mkdir()
            Path(test_dir, "test_sample.py").write_text(
                "import unittest\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(1 + 1, 2)\n",
                encoding="utf-8",
            )
            tools = make_tools(Path(directory))
            result = tools.run_shell("python3 tests/test_sample.py", timeout_seconds=10)
            self.assertTrue(result["ok"], result)
            self.assertIn("OK", result["stderr"])

    def test_search_text(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = make_tools(Path(directory))
            tools.write_file("src/app.py", "def hello():\n    return 'hi'\n")
            result = tools.search_text("hello", path=".", file_glob="*.py")
            self.assertTrue(result["ok"])
            self.assertEqual(result["matches"][0]["path"], "src/app.py")


if __name__ == "__main__":
    unittest.main()
