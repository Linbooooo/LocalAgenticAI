import unittest

from local_agent.context import prepare_messages


class ContextTests(unittest.TestCase):
    def test_prepare_messages_keeps_system_and_recent_messages(self):
        messages = [{"role": "system", "content": "system rules"}]
        for index in range(30):
            messages.append({"role": "user", "content": f"message {index} " + ("x" * 200)})

        packed = prepare_messages(messages, token_budget=600)

        self.assertEqual(packed[0]["role"], "system")
        self.assertIn("Condensed earlier context", packed[1]["content"])
        self.assertIn("message 29", packed[-1]["content"])
        self.assertLess(len(packed), len(messages))

    def test_prepare_messages_leaves_small_context_unchanged(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hello"}]
        self.assertEqual(prepare_messages(messages, token_budget=1000), messages)

    def test_prepare_messages_bounds_single_oversized_protocol_context(self):
        messages = [
            {"role": "system", "content": "base rules"},
            {
                "role": "system",
                "content": "protocol and user request\n" + ("middle context " * 2000) + "\nlatest evidence",
            },
        ]

        packed = prepare_messages(messages, token_budget=1000)

        self.assertEqual(len(packed), 2)
        self.assertIn("protocol and user request", packed[1]["content"])
        self.assertIn("latest evidence", packed[1]["content"])
        self.assertIn("middle context omitted", packed[1]["content"])
        self.assertLess(len(packed[1]["content"]), 4000)


if __name__ == "__main__":
    unittest.main()
