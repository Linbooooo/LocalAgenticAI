import unittest
from pathlib import Path
from unittest.mock import Mock

from local_agent.agent import LocalAgent
from local_agent.config import AgentConfig
from local_agent.ollama_client import OllamaConnectionError


class AdaptiveContextTests(unittest.TestCase):
    def test_memory_error_halves_context_and_retries(self):
        config = AgentConfig(workspace=Path("."), num_ctx=8192, max_num_ctx=8192, min_num_ctx=2048)
        config.finalize()
        agent = LocalAgent(config)
        agent.client.chat = Mock(
            side_effect=[
                OllamaConnectionError("model requires more system memory"),
                {"message": {"role": "assistant", "content": "ok"}},
            ]
        )

        result = agent._chat()

        self.assertEqual(result["message"]["content"], "ok")
        self.assertEqual(agent.config.num_ctx, 4096)
        self.assertEqual(agent.client.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
