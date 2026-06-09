from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from local_agent.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.lines = [json.dumps(event).encode() + b"\n" for event in events]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


class OllamaClientTests(unittest.TestCase):
    def test_stream_collects_content_and_metrics(self) -> None:
        events = [
            {"message": {"content": "hel"}, "done": False},
            {
                "message": {"content": "lo"},
                "done": True,
                "total_duration": 2_000_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 10,
                "eval_duration": 500_000_000,
            },
        ]
        with patch("local_agent.ollama_client.urlopen", return_value=FakeResponse(events)):
            response, metrics = OllamaClient("http://127.0.0.1:11434").chat_stream(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(response["message"]["content"], "hello")
        self.assertEqual(metrics.total_ms, 2000)
        self.assertEqual(metrics.prompt_tps, 20)
        self.assertEqual(metrics.generation_tps, 20)
        self.assertGreaterEqual(metrics.ttft_ms, 0)


if __name__ == "__main__":
    unittest.main()
