import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent.config import AgentConfig


class ConfigTests(unittest.TestCase):
    def test_defaults_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = AgentConfig(workspace=Path(temp))
            config.finalize()
            self.assertEqual(config.model, "qwen2.5-coder:14b")
            self.assertEqual(config.context_budget_tokens(), 3072)

    def test_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            env = {
                "LOCAL_AGENT_WORKSPACE": temp,
                "LOCAL_AGENT_NUM_CTX": "8192",
                "LOCAL_AGENT_NUM_PREDICT": "2048",
                "LOCAL_AGENT_SHELL_TIMEOUT": "30",
            }
            with patch.dict(os.environ, env, clear=False):
                config = AgentConfig.load(None)
            config.finalize()
            self.assertEqual(config.context_budget_tokens(), 6144)
            self.assertEqual(config.shell_timeout, 30)

    def test_rejects_remote_ollama(self) -> None:
        config = AgentConfig(ollama_url="https://example.com")
        with self.assertRaises(ValueError):
            config.finalize()


if __name__ == "__main__":
    unittest.main()
