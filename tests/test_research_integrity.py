"""Regression checks for scientific correctness; no dataset downloads required."""

import contextlib
import io
import json
from pathlib import Path
import random
import statistics
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks import SplitCIFAR10, SplitCIFAR100, PermutedMNIST
from models.mcn import MCN, TaskModule
from models.mcn_v2 import MCNv2
from models.mcn_ablations import MCNBaseOnly, MCNNoGate, MCNNoRouter
from trainers.mcn_trainer import MCNTrainer
from utils.evaluation import evaluate_task_free
from utils.experiments import aggregate_runs, seed_everything, write_json
from utils.metrics import MetricTracker


torch.set_num_threads(1)


def train_small_task(model, task_id):
    # A distribution shift exercises BatchNorm state, not just weight freezing.
    x = torch.randn(4, 3, 8, 8) + 3 * task_id
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    trainer = MCNTrainer(model, torch.device("cpu"), epochs_per_task=1)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        trainer.train_task(task_id, loader)


class FrozenStateTests(unittest.TestCase):
    def test_completed_paths_survive_later_training(self):
        for cls in (MCN, MCNv2, MCNNoGate, MCNNoRouter, MCNBaseOnly):
            with self.subTest(model=cls.__name__):
                seed_everything(7)
                kwargs = dict(num_tasks=3, num_classes_per_task=2,
                              base_dim=16, input_size=8)
                if cls is not MCNBaseOnly:
                    kwargs.update(task_dim=8, freeze_all=True)
                model = cls(**kwargs)
                train_small_task(model, 0)
                train_small_task(model, 1)
                x = torch.randn(4, 3, 8, 8)
                model.eval()
                with torch.no_grad():
                    logits = [model(x, task) for task in (0, 1)]
                modules = [*model._base_modules_to_freeze(),
                           *model._task_components(0), *model._task_components(1)]
                states = [{key: val.clone() for key, val in module.state_dict().items()}
                          for module in modules]
                train_small_task(model, 2)
                for module, before in zip(modules, states):
                    self.assertFalse(module.training)
                    self.assertTrue(all(not p.requires_grad for p in module.parameters()))
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                model.eval()
                with torch.no_grad():
                    for task in (0, 1):
                        torch.testing.assert_close(model(x, task), logits[task], rtol=0, atol=0)

    def test_adaptive_high_encoder_remains_trainable(self):
        for cls in (MCN, MCNv2):
            with self.subTest(model=cls.__name__):
                model = cls(2, 2, base_dim=16, task_dim=8, input_size=8, freeze_all=False)
                train_small_task(model, 0)
                low = {key: value.clone() for key, value in model.base_low.state_dict().items()}
                high = next(model.base_high.parameters()).detach().clone()
                train_small_task(model, 1)
                for key, value in model.base_low.state_dict().items():
                    torch.testing.assert_close(value, low[key], rtol=0, atol=0)
                self.assertTrue(model.base_high.training)
                self.assertFalse(torch.equal(high, next(model.base_high.parameters())))

    def test_checkpoint_restores_lifecycle_and_predictions(self):
        for cls in (MCN, MCNv2, MCNNoGate, MCNBaseOnly):
            with self.subTest(model=cls.__name__):
                kwargs = dict(num_tasks=2, num_classes_per_task=2, base_dim=16, input_size=8)
                if cls is not MCNBaseOnly:
                    kwargs.update(task_dim=8, freeze_all=True)
                model = cls(**kwargs)
                train_small_task(model, 0)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = cls(**kwargs)
                with contextlib.redirect_stdout(io.StringIO()):
                    restored.load_state_dict(torch.load(buffer, weights_only=True))
                self.assertEqual(restored._num_trained, 1)
                restored.train()
                self.assertFalse(restored._base_modules_to_freeze()[0].training)
                x = torch.randn(4, 3, 8, 8)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x, 0), restored(x, 0), rtol=0, atol=0)
                if hasattr(restored, "predict_task_free"):
                    a = model.predict_task_free(x)
                    b = restored.predict_task_free(x)
                    for expected, actual in zip(a, b):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TaskFreeTests(unittest.TestCase):
    class WrongHead(torch.nn.Module):
        def predict_task_free(self, x):
            return torch.zeros(len(x), dtype=torch.long), torch.ones(len(x), dtype=torch.long)

    def benchmark_without_downloads(self, cls):
        benchmark = cls.__new__(cls)
        benchmark.num_tasks = 2
        benchmark.get_test_loader = lambda task: [(torch.zeros(3, 1), torch.zeros(3, dtype=torch.long))]
        return benchmark

    def test_wrong_head_cannot_get_credit_for_same_local_cifar_label(self):
        for cls in (SplitCIFAR10, SplitCIFAR100):
            with self.subTest(benchmark=cls.__name__):
                benchmark = self.benchmark_without_downloads(cls)
                result = evaluate_task_free(self.WrongHead(), benchmark, torch.device("cpu"))
                self.assertEqual(result["per_task"][0]["accuracy"], 0.0)
                self.assertEqual(result["per_task"][1]["accuracy"], 1.0)
                self.assertEqual(result["accuracy"], 0.5)
                self.assertEqual(result["task_selection_accuracy"], 0.5)

    def test_shared_mnist_labels_allow_correct_digit_from_another_head(self):
        benchmark = self.benchmark_without_downloads(PermutedMNIST)
        result = evaluate_task_free(self.WrongHead(), benchmark, torch.device("cpu"))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["task_selection_accuracy"], 0.5)

    def test_nonzero_local_labels_map_to_original_cifar_ids(self):
        benchmark = self.benchmark_without_downloads(SplitCIFAR100)
        actual = benchmark.to_global_labels(torch.tensor([4, 2]), torch.tensor([0, 1]))
        self.assertEqual(actual.tolist(), [4, 7])


class AblationTests(unittest.TestCase):
    def test_no_gate_preserves_layers_and_dropout(self):
        seed_everything(12)
        gated = TaskModule(in_channels=3, out_dim=8, input_size=8)
        seed_everything(12)
        ungated = TaskModule(in_channels=3, out_dim=8, input_size=8, use_gate=False)
        self.assertEqual(set(gated.state_dict()) - set(ungated.state_dict()), {"gate"})
        for name, value in ungated.state_dict().items():
            torch.testing.assert_close(value, gated.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(repr(gated.fc), repr(ungated.fc))
        self.assertEqual(sum(p.numel() for p in gated.parameters()),
                         sum(p.numel() for p in ungated.parameters()) + 1)
        gated.eval()
        ungated.eval()
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(gated(x), ungated(x) * gated.gate.sigmoid())

    def test_full_ablation_preserves_paired_initialization(self):
        seed_everything(4)
        full = MCN(2, 2, base_dim=16, task_dim=8, input_size=8)
        seed_everything(4)
        ablation = MCNNoGate(2, 2, base_dim=16, task_dim=8, input_size=8)
        for name, parameter in ablation.named_parameters():
            torch.testing.assert_close(parameter, full.state_dict()[name], rtol=0, atol=0)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_replays_training_after_unrelated_random_work(self):
        def run():
            seed_everything(9, deterministic=True)
            model = MCN(2, 2, base_dim=16, task_dim=8, input_size=8, freeze_all=True)
            train_small_task(model, 0)
            train_small_task(model, 1)
            return model.state_dict(), random.random(), np.random.rand()
        first, py_first, np_first = run()
        torch.randn(30)
        random.random()
        np.random.rand(8)
        second, py_second, np_second = run()
        self.assertEqual(py_first, py_second)
        self.assertEqual(np_first, np_second)
        for key in first:
            if isinstance(first[key], torch.Tensor):
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            else:
                self.assertEqual(first[key], second[key])

    def test_metrics_and_json_preserve_unmeasured_cells(self):
        tracker = MetricTracker(3)
        tracker.R[:] = [[.9, np.nan, np.nan], [.8, .7, np.nan], [.75, .65, .8]]
        self.assertAlmostEqual(tracker.average_accuracy(), 2.2 / 3)
        self.assertAlmostEqual(tracker.backward_transfer(), -.1)
        self.assertAlmostEqual(tracker.forgetting_measure(), .1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            write_json(path, tracker.summary())
            result = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertIsNone(result["accuracy_matrix"][0][1])

    def test_aggregation_uses_sample_std_and_no_fake_single_seed_certainty(self):
        records = [{"method": "mcn", "seed": seed, "metrics": {
            "average_accuracy": aa, "backward_transfer": 0.0,
            "forgetting_measure": 0.0, "task_free": None}}
            for seed, aa in [(1, .8), (2, .9), (3, .7)]]
        summary = aggregate_runs(records)["mcn"]
        self.assertAlmostEqual(summary["average_accuracy"]["mean"], .8)
        self.assertAlmostEqual(summary["average_accuracy"]["std"], statistics.stdev([.8, .9, .7]))
        self.assertIsNone(aggregate_runs(records[:1])["mcn"]["average_accuracy"]["std"])


if __name__ == "__main__":
    unittest.main()
