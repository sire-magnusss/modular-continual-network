"""
HAT Trainer
-----------
Implements the HAT training loop with:
  1. Temperature annealing within each epoch (s: s_low → s_max)
  2. HAT regularization loss added to cross-entropy
  3. Gradient compensation after backward (protecting previous task neurons)
  4. Task completion cleanup (clip embeddings to binary)

Temperature schedule:
  Within each epoch, s is annealed linearly from s_max/100 → s_max
  across the batches. This gives the model time to explore soft masks at
  the start of training before committing to hard binary masks by the end.
  Over multiple epochs this creates a curriculum: early epochs explore,
  late epochs commit.

The key HAT loop (per batch):
  1. Compute s for this step (linear within epoch)
  2. Forward pass with current s → soft masks
  3. CE loss + HAT regularization
  4. Backward
  5. Gradient compensation (scale embedding grads by 1 - cumulative mask)
  6. Optimizer step
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import MetricTracker


class HATTrainer:

    def __init__(self, model, device: torch.device,
                 lr: float = 1e-3, epochs_per_task: int = 5,
                 hat_c: float = 0.75):
        """
        Args:
            hat_c: regularization coefficient for HAT penalty (default 0.75)
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs_per_task = epochs_per_task
        self.hat_c = hat_c
        self.model.to(device)

    def _get_s(self, batch_idx: int, total_batches: int, epoch: int) -> float:
        """
        Temperature schedule: linear ramp from s_low → s_max within each epoch.
        Across epochs, the full ramp happens each epoch (not cumulative).
        This ensures early epochs still explore (start low), late epochs commit.
        """
        s_low = self.model.s_max / 100.0
        # Progress within current epoch: 0.0 → 1.0
        progress = batch_idx / max(total_batches - 1, 1)
        return s_low + (self.model.s_max - s_low) * progress

    def train_task(self, task_id: int, train_loader: DataLoader):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        total_batches = len(train_loader)

        for epoch in range(self.epochs_per_task):
            total_loss = 0.0
            total_ce = 0.0
            total_reg = 0.0
            correct = 0
            total = 0

            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"  [HAT-T{task_id}] Epoch {epoch+1}/{self.epochs_per_task}",
                leave=False,
            )

            for batch_idx, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)

                # Temperature for this batch
                s = self._get_s(batch_idx, total_batches, epoch)

                optimizer.zero_grad()

                # Forward with temperature s → soft masks
                logits = self.model(x, task_id, s=s)
                ce_loss = criterion(logits, y)

                # HAT regularization: penalize reusing past capacity
                current_masks = self.model.get_masks(task_id, s)
                reg_loss = self.model.hat_regularization(task_id, current_masks, c=self.hat_c)

                loss = ce_loss + reg_loss
                loss.backward()

                # Gradient compensation: scale embedding grads by (1 - cumulative mask)
                # This prevents the optimizer from modifying neurons claimed by past tasks
                self.model.compensate_gradients(task_id, current_masks)

                optimizer.step()

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_ce   += ce_loss.item() * batch_size
                total_reg  += (reg_loss.item() if hasattr(reg_loss, 'item') else 0.0) * batch_size
                correct += (logits.argmax(1) == y).sum().item()
                total += batch_size

                pbar.set_postfix(ce=f"{ce_loss.item():.3f}", s=f"{s:.0f}")

            acc = correct / total
            avg_loss = total_loss / total
            avg_reg  = total_reg / total
            print(f"  [HAT-T{task_id}] Epoch {epoch+1}: "
                  f"loss={avg_loss:.3f}  ce={total_ce/total:.3f}  "
                  f"reg={avg_reg:.4f}  acc={acc*100:.1f}%  s={s:.0f}")

        # After training: clip embeddings to binary range, update trained_tasks count
        self.model.complete_task(task_id)

    @torch.no_grad()
    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            # Test with s_max → near-binary masks
            logits = self.model(x, task_id)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return correct / total

    def run(self, benchmark, tracker: MetricTracker):
        total_params = sum(p.numel() for p in self.model.parameters())
        emb_params = sum(e.numel() for e in self.model.embeddings)

        print("\n" + "="*60)
        print("METHOD: HATModel")
        print("="*60)
        print(f"  Total params: {total_params:,}")
        print(f"  Embedding params (masks): {emb_params:,}")
        print(f"  HAT regularization c: {self.hat_c}")
        print(f"  Temperature s_max: {self.model.s_max}")

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
