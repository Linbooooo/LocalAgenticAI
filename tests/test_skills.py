import unittest

from local_agent.skills import format_coding_skills, select_coding_skills


class SkillTests(unittest.TestCase):
    def test_chat_requests_do_not_select_coding_skills(self):
        skills = select_coding_skills("hello there", "chat", ["pyproject.toml"])

        self.assertEqual(skills, [])

    def test_python_test_request_selects_testing_skill(self):
        skills = select_coding_skills(
            "write tests for combination_sum_ii.py and run them",
            "edit",
            ["pyproject.toml", "local_agent/solutions/combination_sum_ii.py", "tests/test_combination_sum_ii.py"],
        )
        names = [skill.name for skill in skills]

        self.assertIn("coding-change", names)
        self.assertIn("project-discovery", names)
        self.assertIn("python-testing", names)
        self.assertIn("algorithm-verification", names)

    def test_failed_observation_selects_debugging_skill(self):
        skills = select_coding_skills(
            "run the tests again",
            "shell",
            ["pyproject.toml", "tests/test_sample.py"],
            [{"result": {"ok": False, "stderr": "ModuleNotFoundError"}}],
        )
        names = [skill.name for skill in skills]

        self.assertIn("debugging", names)

    def test_skill_prompt_contains_procedural_guidance(self):
        skills = select_coding_skills("run tests/test_sample.py", "shell", ["tests/test_sample.py"])
        prompt = format_coding_skills(skills)

        self.assertIn("python-testing", prompt)
        self.assertIn("unittest discover", prompt)

    def test_algorithm_skill_keeps_expected_values_independent(self):
        skills = select_coding_skills("write a python file that implements two sum and test it", "edit", [])
        prompt = format_coding_skills(skills)

        self.assertIn("algorithm-verification", prompt)
        self.assertIn("validity-check helper or brute-force expected result", prompt)
        self.assertIn("keep expected values stable during repair", prompt)
        self.assertIn("independent oracle", prompt)


if __name__ == "__main__":
    unittest.main()
