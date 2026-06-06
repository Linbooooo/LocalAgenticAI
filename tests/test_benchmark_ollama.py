import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_ollama import (
    RunMetrics,
    _tokens_per_second,
    append_markdown_row,
    detect_processor,
    markdown_row,
    summarize,
)


class BenchmarkTests(unittest.TestCase):
    def test_tokens_per_second_uses_nanoseconds(self):
        self.assertEqual(_tokens_per_second(100, 2_000_000_000), 50.0)
        self.assertEqual(_tokens_per_second(0, 2_000_000_000), 0.0)

    def test_detect_processor(self):
        self.assertEqual(
            detect_processor({"models": [{"name": "model", "size": 100, "size_vram": 100}]}, "model"),
            "gpu",
        )
        self.assertEqual(
            detect_processor({"models": [{"name": "model", "size": 100, "size_vram": 0}]}, "model"),
            "cpu",
        )
        self.assertEqual(
            detect_processor({"models": [{"name": "model", "size": 100, "size_vram": 50}]}, "model"),
            "mixed",
        )

    def test_summary_and_markdown_row(self):
        runs = [
            RunMetrics(100, 1000, 10, 20, 200, 50, 25),
            RunMetrics(200, 1200, 20, 20, 220, 60, 30),
            RunMetrics(300, 1400, 30, 20, 240, 70, 35),
        ]

        result = summarize(
            runs,
            model="model",
            processor="gpu",
            ollama_version="1.0",
            label="baseline",
            num_ctx=4096,
            num_predict=128,
            prompt="prompt",
        )

        self.assertEqual(result["ttft_ms_median"], 200)
        self.assertEqual(result["generation_tps_median"], 30)
        self.assertIn("| baseline | gpu | `model` | 3 | 200.00 | 30.00 |", markdown_row(result))

    def test_append_markdown_row_inserts_before_marker(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "performance.md")
            path.write_text("header\n<!-- benchmark-results-end -->\nfooter\n", encoding="utf-8")

            append_markdown_row(path, "| result |")

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "header\n| result |\n<!-- benchmark-results-end -->\nfooter\n",
            )


if __name__ == "__main__":
    unittest.main()
