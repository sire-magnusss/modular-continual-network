"""Trainer for the PackNet baseline."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.packnet import PackNetModel
from utils.metrics import MetricTracker


class PackNetTrainer:
    def __init__(self, model: PackNetModel, device: torch.device,
                 lr: float = 1e-3, epochs_phase1: int = 5,
                 epochs_phase2: int = 2):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs_phase1 = epochs_phase1
        self.epochs_phase2 = epochs_phase2
        self.model.to(device)

    def _move_masks_to_device(self):
        for name in self.model._frozen_masks:
            self.model._frozen_masks[name] = self.model._frozen_masks[name].to(self.device)

    def _train_phase(self, task_id: int, train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer, epochs: int, phase: str):
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader,
                        desc=f"  [PackNet] Task {task_id} {phase} Epoch {epoch+1}/{epochs}",
                        leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, task_id)
                loss = criterion(logits, y)
                loss.backward()

                self.model.freeze_gradients()
                optimizer.step()

                self.model.apply_masks()

                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            acc = correct / total
            print(f"  [PackNet] Task {task_id} {phase} Epoch {epoch+1}: "
                  f"loss={total_loss/total:.3f}  acc={acc*100:.1f}%")

    def train_task(self, task_id: int, train_loader: DataLoader):
        self._move_masks_to_device()

        print(f"  Phase 1: Training free weights...")
        all_params = list(self.model.backbone.parameters())
        opt1 = torch.optim.Adam(all_params, lr=self.lr)
        self._train_phase(task_id, train_loader, opt1, self.epochs_phase1, "Phase1")

        self.model.prune_and_freeze(task_id)

        print(f"  Phase 2: Retraining pruned weights only...")
        free_params = [p for _, p in self.model.get_free_parameters()]
        if free_params:
            opt2 = torch.optim.Adam(free_params, lr=self.lr * 0.5)
            self._train_phase(task_id, train_loader, opt2, self.epochs_phase2, "Phase2")
        else:
            print("  [PackNet] No free weights remaining for Phase 2.")

    @torch.no_grad()
    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x, task_id)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return correct / total

    def run(self, benchmark, tracker: MetricTracker):
        print("\n" + "="*60)
        print(f"METHOD: PackNet (prune_fraction={self.model.prune_fraction})")
        print("="*60)

        for task_id in range(benchmark.num_tasks):
            print(f"\n>>> Training {benchmark.task_description(task_id)}")
            train_loader = benchmark.get_train_loader(task_id)
            self.train_task(task_id, train_loader)

            print(f"  Evaluating all tasks after training task {task_id}...")
            for eval_task in range(task_id + 1):
                test_loader = benchmark.get_test_loader(eval_task)
                acc = self.evaluate(eval_task, test_loader)
                tracker.update(task_id, eval_task, acc)
                status = "<<< THIS TASK" if eval_task == task_id else "     OLDER TASK"
                print(f"    Task {eval_task}: {acc*100:.1f}% {status}")

        tracker.print_matrix()
        return tracker
