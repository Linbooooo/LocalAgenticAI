import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent.config import AgentConfig


class ConfigTests(unittest.TestCase):
    def test_config_requires_loopback_endpoint(self):
        config = AgentConfig(ollama_url="http://example.com:11434", workspace=Path("."))
        with self.assertRaisesRegex(ValueError, "loopback"):
            config.finalize()

    def test_config_accepts_localhost(self):
        config = AgentConfig(ollama_url="http://localhost:11434", workspace=Path("."))
        config.finalize()
        self.assertTrue(config.workspace.exists())

    def test_config_accepts_compose_ollama_host(self):
        config = AgentConfig(ollama_url="http://ollama:11434", workspace=Path("."))
        config.finalize()
        self.assertTrue(config.workspace.exists())

    def test_config_reads_environment_overrides(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_AGENT_MODEL": "deepseek-coder-v2:16b",
                "LOCAL_AGENT_NUM_CTX": "8192",
                "LOCAL_AGENT_MAX_NUM_CTX": "8192",
                "LOCAL_AGENT_ALLOW_NETWORK_TOOLS": "true",
            },
        ):
            config = AgentConfig.load(None)
        self.assertEqual(config.model, "deepseek-coder-v2:16b")
        self.assertEqual(config.num_ctx, 8192)
        self.assertTrue(config.allow_network_tools)

    def test_config_caps_context(self):
        config = AgentConfig(workspace=Path("."), num_ctx=16384, max_num_ctx=4096)
        config.finalize()
        self.assertEqual(config.num_ctx, 4096)

    def test_config_validates_contract_mode(self):
        config = AgentConfig(workspace=Path("."), contract_mode="fallback")
        config.finalize()
        self.assertEqual(config.contract_mode, "fallback")

        bad = AgentConfig(workspace=Path("."), contract_mode="maybe")
        with self.assertRaisesRegex(ValueError, "contract_mode"):
            bad.finalize()


if __name__ == "__main__":
    unittest.main()
