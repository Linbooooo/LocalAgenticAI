import unittest

from local_agent.agent import LocalAgent


class AgentTests(unittest.TestCase):
    def test_parses_plain_json_tool_call(self):
        calls = LocalAgent._tool_calls_from_content('{"name": "hardware_profile", "arguments": {}}')
        self.assertEqual(calls[0]["function"]["name"], "hardware_profile")

    def test_parses_fenced_json_tool_call(self):
        calls = LocalAgent._tool_calls_from_content(
            '```json\n{"tool_name": "read_file", "args": {"path": "README.md"}}\n```'
        )
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(calls[0]["function"]["arguments"]["path"], "README.md")

    def test_parses_leading_json_tool_call_with_trailing_text(self):
        calls = LocalAgent._tool_calls_from_content(
            '{"name": "hardware_profile", "arguments": {}}\n\nI will use the result next.'
        )
        self.assertEqual(calls[0]["function"]["name"], "hardware_profile")

    def test_ignores_regular_content(self):
        self.assertEqual(LocalAgent._tool_calls_from_content("hello there"), [])


if __name__ == "__main__":
    unittest.main()
