import tempfile
import unittest
from pathlib import Path

from local_agent.config import AgentConfig
from local_agent.tools import ToolRegistry


def make_registry(workspace: Path) -> ToolRegistry:
    config = AgentConfig(workspace=workspace, trust="auto")
    config.finalize()
    return ToolRegistry(config)


class ToolTests(unittest.TestCase):
    def test_file_tools_stay_in_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(Path(directory))
            result = registry.run("write_file", {"path": "hello.txt", "content": "hello\nworld\n"})
            self.assertTrue(result["ok"])

            read = registry.run("read_file", {"path": "hello.txt"})
            self.assertIn("1: hello", read["content"])

            escaped = registry.run("read_file", {"path": "../outside.txt"})
            self.assertFalse(escaped["ok"])
            self.assertIn("escapes workspace", escaped["error"])

    def test_shell_blocks_network_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(Path(directory))
            result = registry.run("run_shell", {"command": "curl https://example.com"})
            self.assertFalse(result["ok"])
            self.assertIn("local-only", result["error"])

    def test_search_text(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(Path(directory))
            registry.run("write_file", {"path": "src/app.py", "content": "def hello():\n    return 'hi'\n"})
            result = registry.run("search_text", {"pattern": "hello", "path": ".", "file_glob": "*.py"})
            self.assertTrue(result["ok"])
            self.assertEqual(result["matches"][0]["path"], "src/app.py")

    def test_file_mutation_hidden_without_edit_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(Path(directory))
            registry.set_current_task("hello there")
            tool_names = {schema["function"]["name"] for schema in registry.schemas()}
            self.assertNotIn("replace_in_file", tool_names)
            result = registry.run(
                "replace_in_file",
                {"path": "README.md", "old": "Local Agentic AI", "new": "Local Intelligent Agent"},
            )
            self.assertFalse(result["ok"])
            self.assertIn("did not explicitly ask", result["error"])

    def test_file_mutation_visible_with_edit_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(Path(directory))
            registry.set_current_task("update the readme")
            tool_names = {schema["function"]["name"] for schema in registry.schemas()}
            self.assertIn("replace_in_file", tool_names)


if __name__ == "__main__":
    unittest.main()
