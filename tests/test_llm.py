import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fahrai_calibration.llm.analysis import budget_route, paired_accuracy, policy_metrics, summarize
from fahrai_calibration.llm.cli import main
from fahrai_calibration.llm.grading import grade, parse_response
from fahrai_calibration.llm.io import read_csv
from fahrai_calibration.llm.tasks import generate
from fahrai_calibration.llm.verification import load_bundle

ROOT = Path(__file__).resolve().parents[1]


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs, cls.tasks, cls.rows = load_bundle(ROOT)

    def test_cli_saves_the_report_it_prints(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "analysis"
            console = io.StringIO()
            with redirect_stdout(console):
                main(["--root", str(ROOT), "analyze", "--output", str(output)])
            saved = json.loads((output / "validation.json").read_text())
            self.assertEqual(saved, json.loads(console.getvalue()))
            self.assertIn("environment", saved)

    def test_all_recorded_outputs_are_regraded_without_missing_assignments(self):
        counts = {
            name: sum(r["answer_exact"] for r in self.rows if r["configuration"] == name)
            for name in self.configs
        }
        self.assertEqual(
            counts,
            {
                "baseline": 185,
                "astra": 288,
                "opus": 288,
                "pro": 288,
                "luna": 282,
                "sonnet": 282,
                "lite": 283,
                "qwen_local": 281,
            },
        )
        self.assertEqual(len(self.rows), 2304)

    def test_reference_solvers_and_sampling_reproduce_all_tasks(self):
        self.assertEqual(
            generate(read_csv(ROOT / "data/population.csv")), list(self.tasks.values())
        )

    def test_grading_does_not_accept_empty_explanations_or_boolean_answers(self):
        task = next(iter(self.tasks.values()))
        answer = {
            "final_answer": task["gold_answer"],
            "explanation": "",
            "source_ids": [task["reference"]["source_id"]],
        }
        self.assertFalse(grade(answer, task)["answer_exact"])
        answer.update(final_answer=True, explanation="A method.")
        self.assertFalse(grade(answer, task)["schema_valid"])

    def test_truncated_response_is_not_rescued_by_valid_json(self):
        row = {
            "provider": "anthropic",
            "finish_reason": "max_tokens",
            "response_text": '{"final_answer":1,"explanation":"x","source_ids":[]}',
        }
        self.assertIsNone(parse_response(row))

    def test_citations_are_checked_separately_from_numeric_success(self):
        task = next(iter(self.tasks.values()))
        score = grade(
            {
                "final_answer": task["gold_answer"],
                "explanation": "A method.",
                "source_ids": ["WRONG"],
            },
            task,
        )
        self.assertTrue(score["answer_exact"])
        self.assertFalse(score["source_ids_valid"])

    def test_budget_ties_share_probability_without_gold_labels(self):
        scores = np.array([0.2, 0.2, 0.7])
        mass = np.array([0.1, 0.3, 0.6])
        route = budget_route(scores, mass, 0.2)
        np.testing.assert_allclose(route, [0.5, 0.5, 0])
        np.testing.assert_allclose(route, budget_route(2 * scores + 0.1, mass, 0.2))
        self.assertAlmostEqual(route @ mass, 0.2)

    def test_model_quality_does_not_change_simulated_learner_need(self):
        route, mass, probability = np.array([0.2, 0.8]), np.array([0.5, 0.5]), np.array([0.1, 0.9])
        good = policy_metrics(route, mass, probability, np.array([[1, 0, 2], [1, 0, 2]]))
        bad = policy_metrics(route, mass, probability, np.array([[0, 0, 2], [0, 0, 2]]))
        self.assertEqual(good[1], bad[1])
        self.assertEqual(good[2], good[0])
        self.assertEqual(bad[2], 0)

    def test_repeats_do_not_double_bootstrap_sample_size(self):
        first = [r for r in self.rows if r["repeat"] == 1]
        once = paired_accuracy(self.configs, self.tasks, first, "test")
        twice = paired_accuracy(self.configs, self.tasks, first + first, "test")
        self.assertEqual(once, twice)
        self.assertTrue(all(row["tasks"] == 144 for row in once))

    def test_zero_local_api_fee_does_not_mean_zero_compute_cost(self):
        local = summarize(self.configs, self.rows, "test")[-1]
        self.assertEqual(local["configuration"], "qwen_local")
        self.assertEqual(local["api_fee_estimate_usd"], 0)
        self.assertIsNone(local["total_compute_cost_usd"])


if __name__ == "__main__":
    unittest.main()
