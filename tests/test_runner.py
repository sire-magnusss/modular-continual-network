"""Exercise the actual runner and checkpoint format with tiny synthetic data."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch.utils.data import DataLoader, TensorDataset

import main
from benchmarks import SplitCIFAR10


class TinyCIFAR(SplitCIFAR10):
    """A plumbing fixture, not CIFAR data and not a benchmark result."""

    def __init__(self):
        self.num_tasks = 2
        self.name = "SyntheticCIFARFixture"
        generator = torch.Generator().manual_seed(81)
        self.datasets = [TensorDataset(torch.randn(4, 3, 32, 32, generator=generator),
                                       torch.tensor([0, 1, 0, 1])) for _ in range(2)]

    def get_train_loader(self, task_id):
        return DataLoader(self.datasets[task_id], batch_size=4, shuffle=True)

    def get_test_loader(self, task_id):
        return DataLoader(self.datasets[task_id], batch_size=4)

    def protocol(self):
        return {"name": self.name, "synthetic": True,
                "task_classes": [[0, 1], [2, 3]], "task_order": [0, 1]}


class RunnerTests(unittest.TestCase):
    def run_fixture(self, output, methods, seeds):
        argv = ["main.py", "--benchmark", "cifar10", "--tasks", "2", "--epochs", "1",
                "--methods", *methods, "--seeds", *map(str, seeds),
                "--device", "cpu", "--threads", "1", "--deterministic", "--task-free",
                "--no-plots", "--output-dir", output]
        with patch("sys.argv", argv), \
                patch("main.build_benchmark", return_value=(TinyCIFAR(), "cifar10")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            main.main()
        return next(Path(output).iterdir())

    def test_multiseed_artifacts_checkpoints_and_method_order_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.run_fixture(str(Path(tmp) / "first"), ["mcn", "mcn_no_gate"], [3, 5])
            summary = json.loads((root / "summary.json").read_text())
            experiment = json.loads((root / "experiment.json").read_text())
            self.assertTrue(experiment["protocol"]["synthetic"])
            self.assertIn("torch", experiment["environment"]["packages"])
            self.assertIn("main.py", experiment["source"]["source_sha256"])
            with zipfile.ZipFile(root / "source.zip") as archive:
                self.assertIn("models/lifecycle.py", archive.namelist())
            for method in ("mcn", "mcn_no_gate"):
                self.assertEqual(summary[method]["n"], 2)
                self.assertIsNotNone(summary[method]["average_accuracy"]["std"])
                for seed in (3, 5):
                    run_dir = root / f"seed_{seed}" / method
                    run = json.loads((run_dir / "run.json").read_text())
                    self.assertEqual(run["status"], "complete")
                    self.assertIsNone(run["metrics"]["accuracy_matrix"][0][1])
                    self.assertEqual(run["metrics"]["task_free"]["label_space"], "global")
                    self.assertGreater((run_dir / "train.log").stat().st_size, 0)
                    checkpoint = torch.load(run_dir / "final_model.pt", weights_only=True)
                    restored = main.build_model(**checkpoint["model_config"])
                    with contextlib.redirect_stdout(io.StringIO()):
                        restored.load_state_dict(checkpoint["state_dict"])
                    self.assertEqual(restored._num_trained, 2)

            other = self.run_fixture(str(Path(tmp) / "reordered"), ["mcn_no_gate", "mcn"], [3])
            for method in ("mcn", "mcn_no_gate"):
                first = root / "seed_3" / method
                second = other / "seed_3" / method
                self.assertEqual(json.loads((first / "run.json").read_text())["metrics"],
                                 json.loads((second / "run.json").read_text())["metrics"])
                a = torch.load(first / "final_model.pt", weights_only=True)["state_dict"]
                b = torch.load(second / "final_model.pt", weights_only=True)["state_dict"]
                for name in a:
                    if isinstance(a[name], torch.Tensor):
                        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_failed_runs_retain_error_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("main.run_method", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    self.run_fixture(tmp, ["mcn"], [3])
            root = next(Path(tmp).iterdir())
            run_dir = root / "seed_3" / "mcn"
            run = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(run["status"], "failed")
            self.assertIn("fixture failure", (run_dir / "train.log").read_text())


if __name__ == "__main__":
    unittest.main()
