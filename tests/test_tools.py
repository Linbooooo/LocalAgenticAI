from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent.config import AgentConfig
from local_agent.tools import WorkspaceShell, blocked_command_reason


class WorkspaceShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def shell(self, **overrides) -> WorkspaceShell:
        config = AgentConfig(workspace=self.workspace, trust="auto", **overrides)
        config.finalize()
        return WorkspaceShell(config)

    def test_runs_from_workspace(self) -> None:
        result = self.shell().run("printf '%s' \"$PWD\"")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, str(self.workspace))

    def test_captures_failure(self) -> None:
        result = self.shell().run("printf problem >&2; exit 7")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stderr, "problem")
        self.assertFalse(result.ok)

    def test_accepts_repeated_y_confirmation(self) -> None:
        config = AgentConfig(workspace=self.workspace, trust="ask")
        config.finalize()
        with patch("builtins.input", return_value="yy"):
            result = WorkspaceShell(config).run("printf yes")
        self.assertTrue(result.ok)

    def test_decline_is_reported(self) -> None:
        config = AgentConfig(workspace=self.workspace, trust="ask")
        config.finalize()
        with patch("builtins.input", return_value="no"):
            result = WorkspaceShell(config).run("printf no")
        self.assertTrue(result.cancelled)

    def test_network_and_destructive_commands_are_blocked(self) -> None:
        self.assertIsNotNone(blocked_command_reason("curl https://example.com", allow_network=False))
        self.assertIsNotNone(blocked_command_reason("rm -rf build", allow_network=False))
        self.assertIsNone(blocked_command_reason("curl https://example.com", allow_network=True))


if __name__ == "__main__":
    unittest.main()
