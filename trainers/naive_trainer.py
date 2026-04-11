"""
Naive Sequential Trainer
-------------------------
Just trains on each task one at a time with standard SGD/Adam.
No mechanism to prevent forgetting.

This is your baseline that WILL catastrophically forget.
You'll see Task 0 accuracy collapse to near-random after training Task 1.
That collapse is the problem everything else is trying to solve.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import MetricTracker


class NaiveTrainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, epochs_per_task: int = 5):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.model.to(device)

    def train_task(self, task_id: int, train_loader: DataLoader):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs_per_task):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"  [Naive] Task {task_id} Epoch {epoch+1}/{self.epochs_per_task}",
                        leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, task_id)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            acc = correct / total
            print(f"  [Naive] Task {task_id} Epoch {epoch+1}: "
                  f"loss={total_loss/total:.3f}  acc={acc*100:.1f}%")

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
        """Full continual learning loop over all tasks."""
        print("\n" + "="*60)
        print("METHOD: Naive (no forgetting prevention)")
        print("="*60)

        for task_id in range(benchmark.num_tasks):
            print(f"\n>>> Training {benchmark.task_description(task_id)}")
            train_loader = benchmark.get_train_loader(task_id)
            self.train_task(task_id, train_loader)

            # Evaluate on ALL tasks seen so far
            print(f"  Evaluating all tasks after training task {task_id}...")
            for eval_task in range(task_id + 1):
                test_loader = benchmark.get_test_loader(eval_task)
                acc = self.evaluate(eval_task, test_loader)
                tracker.update(task_id, eval_task, acc)
                status = "<<< THIS TASK" if eval_task == task_id else "     OLDER TASK"
                print(f"    Task {eval_task}: {acc*100:.1f}% {status}")

        tracker.print_matrix()
        return tracker
