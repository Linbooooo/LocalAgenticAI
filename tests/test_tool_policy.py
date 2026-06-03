import unittest

from local_agent.tool_policy import extract_direct_shell_command


class PolicyTests(unittest.TestCase):
    def test_extracts_quoted_direct_shell_command(self):
        self.assertEqual(extract_direct_shell_command('execute "nvidia-smi"'), "nvidia-smi")

    def test_does_not_extract_prose_shell_request(self):
        self.assertIsNone(extract_direct_shell_command("run the file for me"))


if __name__ == "__main__":
    unittest.main()
