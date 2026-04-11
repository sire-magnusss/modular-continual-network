"""
MCN Trainer
------------
Training protocol for the Modular Continual Network:

  Task 0:
    - Train ALL parameters (base encoder + task 0 module/router/head)
    - After training: FREEZE the base encoder

  Task t > 0:
    - Only train task module[t] + router[t] + head[t]
    - Base encoder is frozen — Task 0 representations are untouched
    - No penalties, no masks — isolation is architectural

This is the key claim to validate:
  "With a frozen base encoder and task-specific adapters, we can achieve
   near-zero forgetting while maintaining plasticity for new tasks."
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.mcn import MCN
from utils.metrics import MetricTracker


class MCNTrainer:
    def __init__(self, model: MCN, device: torch.device,
                 lr: float = 1e-3, epochs_per_task: int = 5):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.model.to(device)

    def train_task(self, task_id: int, train_loader: DataLoader):
        self.model.train()

        # Get only the parameters relevant to this task
        params = self.model.get_task_parameters(task_id)
        optimizer = torch.optim.Adam(params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        is_task_zero = not self.model._base_frozen
        prefix = "[MCN-T0]" if is_task_zero else f"[MCN-T{task_id}]"

        for epoch in range(self.epochs_per_task):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader,
                        desc=f"  {prefix} Epoch {epoch+1}/{self.epochs_per_task}",
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
            print(f"  {prefix} Epoch {epoch+1}: loss={total_loss/total:.3f}  acc={acc*100:.1f}%")

        # After Task 0, freeze the base encoder
        if task_id == 0 and not self.model._base_frozen:
            self.model.freeze_base_encoder()

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
        method_name = type(self.model).__name__
        print("\n" + "="*60)
        print(f"METHOD: {method_name}")
        print("="*60)

        # Print parameter budget (only if the model supports it)
        if hasattr(self.model, "param_count"):
            counts = self.model.param_count()
            print(f"  Base encoder params : {counts['base_encoder']:,}")
            print(f"  Per-task params     : {counts['per_task_total']:,} "
                  f"(module={counts['per_task_module']:,}, "
                  f"router={counts['per_task_router']:,}, "
                  f"head={counts['per_task_head']:,})")
            print(f"  Total ({benchmark.num_tasks} tasks)    : {counts['total_for_n_tasks']:,}")
        else:
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Total params: {total:,}")

        for task_id in range(benchmark.num_tasks):
            print(f"\n>>> Training {benchmark.task_description(task_id)}")
            if task_id == 0:
                print(f"  [Task 0] Training full network, then freezing base encoder")
            else:
                print(f"  [Task {task_id}] Training only: task module + router + head")

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
