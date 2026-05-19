import unittest

from local_agent.tool_policy import classify_tool_policy


class ToolPolicyTests(unittest.TestCase):
    def test_greeting_has_no_tools(self):
        policy = classify_tool_policy("hello there")
        self.assertEqual(policy.name, "chat")
        self.assertEqual(policy.allowed_tools, frozenset())

    def test_repo_inspection_is_read_only(self):
        policy = classify_tool_policy("inspect this repository")
        self.assertEqual(policy.name, "read")
        self.assertTrue(policy.allows("read_file"))
        self.assertFalse(policy.allows("replace_in_file"))

    def test_edit_request_allows_mutation(self):
        policy = classify_tool_policy("update the README")
        self.assertEqual(policy.name, "edit")
        self.assertTrue(policy.allows("replace_in_file"))


if __name__ == "__main__":
    unittest.main()
