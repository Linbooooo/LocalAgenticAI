import unittest

from local_agent.context import prepare_messages


class ContextTests(unittest.TestCase):
    def test_small_history_is_unchanged(self) -> None:
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]
        self.assertIs(prepare_messages(messages, 100), messages)

    def test_keeps_system_prompt_and_recent_messages(self) -> None:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": str(index) * 200} for index in range(8))

        packed = prepare_messages(messages, 180)

        self.assertEqual(packed[0], messages[0])
        self.assertIn("earlier messages were omitted", packed[1]["content"])
        self.assertEqual(packed[-1], messages[-1])
        self.assertLess(len(packed), len(messages))


if __name__ == "__main__":
    unittest.main()
