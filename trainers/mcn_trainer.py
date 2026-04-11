"""
MCN Trainer — with Adaptive Layer Freezing support
----------------------------------------------------
Task 0:
  - Train ALL parameters at base_lr
  - After training: freeze base_low, leave base_high adaptable

Task t > 0:
  - base_high params  at base_lr × adaptive_lr_scale  (slow drift OK)
  - task module + router + head  at base_lr            (full plasticity)

The two-group optimizer lets base_high slowly adapt to new task domains
without aggressively forgetting Task 0 structure. The key insight is that
high-level semantic representations need to shift for structurally different
tasks (e.g., permuted MNIST), but should move slowly to preserve past learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import MetricTracker


class MCNTrainer:
    def __init__(self, model, device: torch.device,
                 lr: float = 1e-3, epochs_per_task: int = 5):
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.model.to(device)

    def _build_optimizer(self, task_id: int) -> torch.optim.Optimizer:
        """
        Build optimizer with appropriate parameter groups.
        Uses get_task_param_groups if available (MCN with adaptive freezing),
        otherwise falls back to get_task_parameters (ablation variants).
        """
        if hasattr(self.model, "get_task_param_groups"):
            param_groups = self.model.get_task_param_groups(task_id, base_lr=self.lr)
        else:
            param_groups = [{"params": self.model.get_task_parameters(task_id),
                             "lr": self.lr}]
        return torch.optim.Adam(param_groups)

    def train_task(self, task_id: int, train_loader: DataLoader):
        self.model.train()
        optimizer = self._build_optimizer(task_id)
        criterion = nn.CrossEntropyLoss()

        is_task_zero = not self.model._base_frozen
        prefix = "[MCN-T0]" if is_task_zero else f"[MCN-T{task_id}]"

        # Show which lr groups are active this task
        if not is_task_zero and hasattr(self.model, "adaptive_lr_scale"):
            scale = self.model.adaptive_lr_scale
            print(f"  lr groups: base_high={self.lr * scale:.2e}  task={self.lr:.2e}")

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

        # After Task 0, apply adaptive freezing
        if task_id == 0 and not self.model._base_frozen:
            self.model.freeze_base_encoder()

        # Track how many tasks have been trained (enables task-free inference)
        if hasattr(self.model, '_num_trained'):
            self.model._num_trained = task_id + 1

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

        if hasattr(self.model, "param_count"):
            counts = self.model.param_count()
            # Show split if adaptive freezing is present
            if "base_low" in counts:
                print(f"  base_low  (frozen)   : {counts['base_low']:,} params")
                print(f"  base_high (adaptive) : {counts['base_high']:,} params  "
                      f"@ {getattr(self.model,'adaptive_lr_scale',1.0)}× lr")
            else:
                print(f"  Base encoder params  : {counts['base_encoder']:,}")
            extra = ""
            if "per_task_cross_attn" in counts:
                extra = f", cross_attn={counts['per_task_cross_attn']:,}"
            print(f"  Per-task params      : {counts['per_task_total']:,} "
                  f"(module={counts['per_task_module']:,}"
                  f"{extra}, "
                  f"router={counts['per_task_router']:,}, "
                  f"head={counts['per_task_head']:,})")
            print(f"  Total ({benchmark.num_tasks} tasks)     : {counts['total_for_n_tasks']:,}")
        else:
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Total params: {total:,}")

        for task_id in range(benchmark.num_tasks):
            print(f"\n>>> Training {benchmark.task_description(task_id)}")
            if task_id == 0:
                print(f"  [Task 0] Training full network")
            else:
                print(f"  [Task {task_id}] task module + router + head (full lr) "
                      f"+ base_high (adaptive lr)")

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

        # Task-free evaluation (if model supports it)
        if hasattr(self.model, 'predict_task_free'):
            print("\n--- Task-Free Inference (entropy-based task selection) ---")
            correct_total = 0
            total_samples = 0
            for eval_task in range(benchmark.num_tasks):
                test_loader = benchmark.get_test_loader(eval_task)
                correct = 0
                total = 0
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred_labels, pred_tasks = self.model.predict_task_free(x)
                    correct += (pred_labels == y).sum().item()
                    total += x.size(0)
                acc = correct / total
                correct_total += correct
                total_samples += total
                print(f"  Task {eval_task} (task-free): {acc*100:.1f}%")
            overall = correct_total / total_samples
            print(f"  Overall task-free accuracy: {overall*100:.1f}%")

        return tracker
