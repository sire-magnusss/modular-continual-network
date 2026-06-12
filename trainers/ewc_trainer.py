"""Trainer for the EWC baseline."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ewc import EWCModel
from utils.metrics import MetricTracker


class EWCTrainer:
    def __init__(self, model: EWCModel, device: torch.device,
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
            total_ewc = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"  [EWC] Task {task_id} Epoch {epoch+1}/{self.epochs_per_task}",
                        leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits = self.model(x, task_id)
                task_loss = criterion(logits, y)
                ewc_loss = self.model.ewc_penalty().to(self.device)
                loss = task_loss + ewc_loss

                loss.backward()
                optimizer.step()

                total_loss += task_loss.item() * x.size(0)
                total_ewc += ewc_loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
                pbar.set_postfix(task=f"{task_loss.item():.3f}",
                                 ewc=f"{ewc_loss.item():.3f}")

            acc = correct / total
            print(f"  [EWC] Task {task_id} Epoch {epoch+1}: "
                  f"task_loss={total_loss/total:.3f}  ewc_loss={total_ewc/total:.4f}  acc={acc*100:.1f}%")

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
        print(f"METHOD: EWC (lambda={self.model.ewc_lambda})")
        print("="*60)

        for task_id in range(benchmark.num_tasks):
            print(f"\n>>> Training {benchmark.task_description(task_id)}")
            train_loader = benchmark.get_train_loader(task_id)
            self.train_task(task_id, train_loader)

            self.model.consolidate(task_id, train_loader, self.device)

            print(f"  Evaluating all tasks after training task {task_id}...")
            for eval_task in range(task_id + 1):
                test_loader = benchmark.get_test_loader(eval_task)
                acc = self.evaluate(eval_task, test_loader)
                tracker.update(task_id, eval_task, acc)
                status = "<<< THIS TASK" if eval_task == task_id else "     OLDER TASK"
                print(f"    Task {eval_task}: {acc*100:.1f}% {status}")

        tracker.print_matrix()
        return tracker
