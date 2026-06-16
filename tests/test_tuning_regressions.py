import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseframework.datahandler import load_data
from baseframework.stream_clustering import OfflinePhase_Clusterer
from baseframework.tuning.engine import run_search
from baseframework.tuning.types import EvaluationRecord, ObjectiveResult
from src.eval.densoinn_eval import reconstruct_point_labels


class _DummyOfflinePhase(OfflinePhase_Clusterer):
    def predict_summaries(self, summaries: dict) -> dict:
        return dict(summaries)


class TuningRegressionTests(unittest.TestCase):
    def test_predict_assignment_treats_missing_summary_ids_as_noise(self):
        phase = _DummyOfflinePhase()
        labels = phase.predict_assignment([1, 99, -1], {1: 7})
        self.assertEqual(labels, [7, -1, -1])

    def test_summary_lookup_falls_back_to_noise_for_deleted_nodes(self):
        labels = reconstruct_point_labels(
            np.zeros((2, 1)),
            {},
            {1: 3},
            assignments=[1, 42],
            label_mode="summary_lookup",
        )
        self.assertEqual(labels, [3, -1])

    def test_run_search_continues_after_candidate_failure(self):
        def evaluator(value: int) -> EvaluationRecord:
            if value < 0:
                raise ValueError("bad candidate")
            return EvaluationRecord(params={"value": value})

        def objective(record: EvaluationRecord) -> ObjectiveResult:
            return ObjectiveResult(name="dummy", score=float(record.params["value"]), valid=True, reason="ok")

        results = run_search(
            evaluator,
            objective,
            [{"value": -1}, {"value": 2}],
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].valid)
        self.assertEqual(results[0].score, 2.0)
        self.assertFalse(results[1].valid)
        self.assertIn("ValueError", results[1].reason)

    def test_run_search_fail_fast_preserves_debug_mode(self):
        def evaluator(value: int) -> EvaluationRecord:
            if value < 0:
                raise ValueError("bad candidate")
            return EvaluationRecord(params={"value": value})

        def objective(record: EvaluationRecord) -> ObjectiveResult:
            return ObjectiveResult(name="dummy", score=float(record.params["value"]), valid=True, reason="ok")

        with self.assertRaises(ValueError):
            run_search(
                evaluator,
                objective,
                [{"value": -1}],
                fail_fast=True,
            )

    def test_load_data_can_skip_labels_for_unsupervised_runs(self):
        X, y = load_data("complex9", batch=1, batch_max=8, scale=False, include_labels=False)
        self.assertEqual(len(X), 8)
        self.assertIsNone(y)


if __name__ == "__main__":
    unittest.main()
