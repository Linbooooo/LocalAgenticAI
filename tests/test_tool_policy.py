import unittest

from local_agent.tool_policy import classify_intent, extract_direct_shell_command


class PolicyTests(unittest.TestCase):
    def test_greeting_is_chat(self):
        self.assertEqual(classify_intent("hello there"), "chat")

    def test_repo_inspection_is_read(self):
        self.assertEqual(classify_intent("inspect this repository"), "read")
        self.assertEqual(classify_intent("display your solution"), "read")

    def test_edit_request_is_edit(self):
        self.assertEqual(classify_intent("update the README"), "edit")

    def test_extracts_quoted_direct_shell_command(self):
        self.assertEqual(extract_direct_shell_command('execute "nvidia-smi"'), "nvidia-smi")

    def test_does_not_extract_prose_shell_request(self):
        self.assertIsNone(extract_direct_shell_command("run the file for me"))


if __name__ == "__main__":
    unittest.main()
