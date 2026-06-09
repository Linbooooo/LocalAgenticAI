from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from local_agent.agent import LocalAgent, extract_shell_command
from local_agent.config import AgentConfig
from local_agent.ollama_client import ModelMetrics
from local_agent.tools import ShellResult


METRICS = ModelMetrics(
    ttft_ms=10,
    total_ms=50,
    prompt_tokens=20,
    output_tokens=10,
    prompt_tps=100,
    generation_tps=20,
)


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[dict[str, object]]] = []

    def chat_stream(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        return {"message": {"content": next(self.responses)}}, METRICS


class FakeShell:
    def __init__(self, results: list[ShellResult]) -> None:
        self.results = iter(results)
        self.commands: list[str] = []

    def run(self, command: str) -> ShellResult:
        self.commands.append(command)
        return next(self.results)


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.config = AgentConfig(workspace=Path(self.temp.name), trust="auto", max_steps=5)
        self.config.finalize()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plain_text_is_a_final_answer(self) -> None:
        client = FakeClient(["Hello."])
        shell = FakeShell([])

        result = LocalAgent(self.config, client=client, shell=shell).run("hello")

        self.assertEqual(result.content, "Hello.")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.commands, 0)
        self.assertIsNone(result.time_to_first_shell_ms)

    def test_shell_result_is_observed_before_finishing(self) -> None:
        client = FakeClient(
            [
                "```bash\npython3 hello.py\n```",
                "The script ran and printed Hello.",
            ]
        )
        shell = FakeShell([ShellResult(returncode=0, stdout="Hello")])

        result = LocalAgent(self.config, client=client, shell=shell).run("run hello.py")

        self.assertEqual(shell.commands, ["python3 hello.py"])
        self.assertEqual(result.commands, 1)
        self.assertIn("printed Hello", result.content)
        second_call = client.calls[1]
        self.assertIn("exit_code: 0", str(second_call[-1]["content"]))
        self.assertIn("stdout:\nHello", str(second_call[-1]["content"]))

    def test_model_can_repair_after_a_failed_command(self) -> None:
        client = FakeClient(
            [
                "```bash\npython3 broken.py\n```",
                "```bash\nsed -i 's/broken/fixed/' broken.py && python3 broken.py\n```",
                "Fixed and verified.",
            ]
        )
        shell = FakeShell(
            [
                ShellResult(returncode=1, stderr="NameError"),
                ShellResult(returncode=0, stdout="fixed"),
            ]
        )

        result = LocalAgent(self.config, client=client, shell=shell).run("fix broken.py")

        self.assertEqual(result.turns, 3)
        self.assertEqual(result.commands, 2)
        self.assertEqual(result.content, "Fixed and verified.")

    def test_declined_command_stops_without_reprompting(self) -> None:
        client = FakeClient(["```bash\ncat > hello.py <<'EOF'\nprint('hello')\nEOF\n```"])
        shell = FakeShell([ShellResult(cancelled=True)])

        result = LocalAgent(self.config, client=client, shell=shell).run("write hello.py")

        self.assertEqual(result.content, "Shell command was not run because approval was declined.")
        self.assertEqual(len(client.calls), 1)

    def test_metrics_are_returned_for_benchmarking(self) -> None:
        result = LocalAgent(
            self.config,
            client=FakeClient(["done"]),
            shell=FakeShell([]),
        ).run("task")

        self.assertEqual(result.model_metrics, [METRICS])
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_new_task_continues_in_the_same_linear_history(self) -> None:
        client = FakeClient(
            [
                "```bash\nprintf hello\n```",
                "First task complete.",
                "```bash\nprintf second\n```",
                "Second task complete.",
            ]
        )
        shell = FakeShell(
            [
                ShellResult(returncode=0, stdout="hello"),
                ShellResult(returncode=0, stdout="second"),
            ]
        )
        agent = LocalAgent(self.config, client=client, shell=shell)

        first = agent.run("run the first command")
        second = agent.run("now run a different command")

        self.assertEqual(first.content, "First task complete.")
        self.assertEqual(second.content, "Second task complete.")
        self.assertEqual(shell.commands, ["printf hello", "printf second"])
        self.assertIn("now run a different command", str(client.calls[2]))


class ShellExtractionTests(unittest.TestCase):
    def test_extracts_shell_blocks_only(self) -> None:
        content = "```python\nprint('no')\n```\n```bash\npython3 app.py\n```"
        self.assertEqual(extract_shell_command(content), "python3 app.py")

    def test_executes_only_the_first_shell_block(self) -> None:
        content = "```sh\nls\n```\nthen\n```shell\npytest\n```"
        self.assertEqual(extract_shell_command(content), "ls")


if __name__ == "__main__":
    unittest.main()
