"""Check training objectives, validation selection, deployment and safe resume."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rice_disease.benchmark import FocalLoss, create_loss, create_optimizer
from rice_disease.data import ROOT, write_json
from rice_disease.tune import activate, candidate_key, trial_configs, tune


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/benchmark.json").read_text())
        torch.set_num_threads(2)

    def test_focal_gamma_zero_matches_cross_entropy_value_and_gradient(self):
        logits = torch.tensor([[1., -2., .5], [-1., 2., 3.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        focal = FocalLoss(0)(logits, labels)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(focal, ce)
        torch.testing.assert_close(torch.autograd.grad(focal, logits)[0],
                                   torch.autograd.grad(ce, logits)[0])
        extreme = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
        loss = FocalLoss(2)(extreme, torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_all_recipes_update_parameters_and_keep_baseline_unchanged(self):
        original = dict(self.config)
        recipes = trial_configs(self.config)
        self.assertEqual(len(recipes), 12)
        self.assertEqual(self.config, original)
        for name, config in recipes:
            with self.subTest(recipe=name):
                head = torch.nn.Linear(3, 6)
                initial = head.weight.detach().clone()
                optimizer = create_optimizer(head.parameters(), config)
                self.assertEqual(type(optimizer).__name__, config["optimizer"])
                loss = create_loss(config)(head(torch.ones(2, 3)), torch.tensor([0, 1]))
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(initial, head.weight))
                self.assertEqual(config["seed"], original["seed"])
                self.assertEqual(config["epochs"], original["epochs"])
        smoothed = create_loss(dict(self.config, loss="LabelSmoothedCrossEntropy", label_smoothing=.1))
        self.assertEqual(smoothed.label_smoothing, .1)
        for key, factory in (("loss", create_loss), ("optimizer", lambda cfg: create_optimizer(head.parameters(), cfg))):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                factory(dict(self.config, **{key: "typo"}))

    def test_ranking_uses_validation_f1_then_size_and_stable_ties(self):
        def candidate(name, f1, size, accuracy):
            return {"run": name, "metrics": {"validation": {"macro_f1": f1, "accuracy": accuracy},
                                              "total_parameters": size}}
        baseline = candidate("baseline", .9, 10, .9)
        tied = candidate("tied", .9, 10, .99)
        better = candidate("better", .91, 100, .8)
        smaller = candidate("smaller", .9, 5, .8)
        result = sorted([baseline, tied, better, smaller], key=candidate_key)
        self.assertEqual([c["run"] for c in result], ["better", "smaller", "baseline", "tied"])

    def test_search_exports_winner_preserves_old_selection_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            artifacts = Path(temp)
            baseline = artifacts / "runs/frozen_baseline"
            config = dict(self.config, models=["tiny"])
            split = {"manifest_sha256": "fixture-split"}
            write_json(baseline / "config.json", config)

            def save_model(name, cfg, run, score):
                result = {"architecture": name, "validation": {"macro_f1": score, "accuracy": score},
                          "train": {"accuracy": .95}, "total_parameters": 10, "best_epoch": 1}
                torch.save({"architecture": name, "config": cfg, "split_sha256": "fixture-split",
                            "validation": result["validation"]}, run / f"{name}.pt")
                write_json(run / f"{name}_metrics.json", result)
                return result

            save_model("tiny", config, baseline, .8)
            old = {"checkpoint": "old-model.pt", "note": "original selection"}
            write_json(artifacts / "selected_model.json", old)
            old_test = artifacts / "final_evaluation/metrics.json"
            write_json(old_test, {"test": "historical"})
            old_bytes = old_test.read_bytes()
            variant = dict(config, loss="FocalLoss")

            def train(name, cfg, rows, info, root, run, **kwargs):
                self.assertEqual(kwargs["feature_config"], config)
                return save_model(name, cfg, run, .9)

            with patch("rice_disease.tune.load_manifest", return_value=([], split)), \
                    patch("rice_disease.tune.trial_configs", return_value=[("focal", variant)]), \
                    patch("rice_disease.tune.summarize"), \
                    patch("rice_disease.tune.benchmark_one", side_effect=train) as training:
                selected = tune(artifacts, verbose=0, experiment="fixture")
                self.assertEqual(selected["config"]["loss"], "FocalLoss")
                self.assertEqual(selected["candidate_count"], 2)
                final = artifacts / selected["checkpoint"]
                self.assertTrue(final.is_file())
                self.assertEqual(hashlib.sha256(final.read_bytes()).hexdigest(), selected["checkpoint_sha256"])
                self.assertEqual(json.loads(next((artifacts / "selection_history").glob("*.json")).read_text()), old)
                self.assertEqual(old_test.read_bytes(), old_bytes)
                self.assertEqual(tune(artifacts, verbose=0, experiment="fixture"), selected)
                self.assertEqual(training.call_count, 1)
                final.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint has changed"):
                    activate(artifacts, selected)


if __name__ == "__main__":
    unittest.main()
