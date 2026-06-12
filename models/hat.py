"""Hard Attention to the Task baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# CIFAR HAT model

class HATModel(nn.Module):
    """CNN HAT model with task-specific output heads."""

    def __init__(self, num_tasks: int, num_classes_per_task: int = 2,
                 in_channels: int = 3, input_size: int = 32,
                 s_max: float = 400.0):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.s_max = s_max

        pooled = input_size // 8

        # Split blocks make mask injection explicit.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pooled * pooled, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task)
            for _ in range(num_tasks)
        ])

        # One task embedding per masked layer.
        self.mask_dims = [64, 128, 256, 512]
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(num_tasks, d))
            for d in self.mask_dims
        ])

        # How many tasks have been fully trained (used for cumulative mask)
        self._trained_tasks = 0

    # Mask utilities

    def get_masks(self, task_id: int, s: float) -> List[torch.Tensor]:
        """Compute soft masks at temperature s for task_id."""
        return [
            torch.sigmoid(s * self.embeddings[l][task_id])
            for l in range(len(self.mask_dims))
        ]

    @torch.no_grad()
    def get_cumulative_mask(self, device=None) -> List[torch.Tensor]:
        """Return the max mask over all completed tasks."""
        if self._trained_tasks == 0:
            return [torch.zeros(d) for d in self.mask_dims]

        s_test = self.s_max
        cum = []
        for l in range(len(self.mask_dims)):
            task_masks = torch.stack([
                torch.sigmoid(s_test * self.embeddings[l][t])
                for t in range(self._trained_tasks)
            ])  # (trained_tasks, D_l)
            cum.append(task_masks.max(0).values)

        if device is not None:
            cum = [c.to(device) for c in cum]
        return cum

    # Forward pass

    def forward(self, x: torch.Tensor, task_id: int,
                s: Optional[float] = None) -> torch.Tensor:
        """Forward pass using the selected task mask."""
        if s is None:
            s = self.s_max

        masks = self.get_masks(task_id, s)

        h = self.conv1(x)
        h = h * masks[0].view(1, -1, 1, 1)

        h = self.conv2(h)
        h = h * masks[1].view(1, -1, 1, 1)

        h = self.conv3(h)
        h = h * masks[2].view(1, -1, 1, 1)

        h = self.fc_block(h)
        h = h * masks[3]

        return self.heads[task_id](h)

    # HAT loss

    def hat_regularization(self, task_id: int,
                            current_masks: List[torch.Tensor],
                            c: float = 0.75) -> torch.Tensor:
        """Penalize reuse of units already used by previous tasks."""
        if task_id == 0:
            return torch.zeros(1, device=current_masks[0].device, requires_grad=False).squeeze()

        device = current_masks[0].device
        cum_masks = self.get_cumulative_mask(device=device)
        reg = torch.zeros(1, device=device)
        total = 0
        for cm, tm in zip(cum_masks, current_masks):
            reg = reg + (cm * tm).sum()
            total += cm.numel()
        return c * reg / total

    # Gradient compensation

    def compensate_gradients(self, task_id: int, current_masks: List[torch.Tensor]):
        """Protect mask embeddings used by earlier tasks."""
        if task_id == 0:
            return

        device = self.embeddings[0].device
        cum_masks = self.get_cumulative_mask(device=device)

        with torch.no_grad():
            for l, (emb, cm) in enumerate(zip(self.embeddings, cum_masks)):
                if emb.grad is not None:
                    emb.grad[:task_id].zero_()
                    emb.grad[task_id] *= (1.0 - cm)

    # Post-task operations

    def complete_task(self, task_id: int):
        """Record a completed task and clamp its mask embeddings."""
        with torch.no_grad():
            for emb in self.embeddings:
                emb[task_id].clamp_(-1.0, 1.0)
        self._trained_tasks = task_id + 1
        pct_used = self._capacity_used_pct()
        print(f"[HAT] Task {task_id} complete. "
              f"Cumulative capacity used: {pct_used:.1f}%")

    def _capacity_used_pct(self) -> float:
        """Fraction of neurons claimed by at least one task."""
        if self._trained_tasks == 0:
            return 0.0
        cum = self.get_cumulative_mask()
        total = sum(c.numel() for c in cum)
        used = sum((c > 0.5).sum().item() for c in cum)
        return 100.0 * used / total

    def get_task_parameters(self, task_id: int):
        return list(self.parameters())


# MNIST HAT model

class HATMNISTModel(nn.Module):
    """MLP HAT model for Permuted MNIST."""

    def __init__(self, num_tasks: int, num_classes_per_task: int = 10,
                 s_max: float = 400.0):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.s_max = s_max

        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([
            nn.Linear(400, num_classes_per_task)
            for _ in range(num_tasks)
        ])

        self.mask_dims = [400, 400]
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(num_tasks, d))
            for d in self.mask_dims
        ])
        self._trained_tasks = 0

    def get_masks(self, task_id: int, s: float) -> List[torch.Tensor]:
        return [
            torch.sigmoid(s * self.embeddings[l][task_id])
            for l in range(len(self.mask_dims))
        ]

    @torch.no_grad()
    def get_cumulative_mask(self, device=None) -> List[torch.Tensor]:
        if self._trained_tasks == 0:
            return [torch.zeros(d) for d in self.mask_dims]
        s_test = self.s_max
        cum = []
        for l in range(len(self.mask_dims)):
            task_masks = torch.stack([
                torch.sigmoid(s_test * self.embeddings[l][t])
                for t in range(self._trained_tasks)
            ])
            cum.append(task_masks.max(0).values)
        if device is not None:
            cum = [c.to(device) for c in cum]
        return cum

    def forward(self, x: torch.Tensor, task_id: int,
                s: Optional[float] = None) -> torch.Tensor:
        if s is None:
            s = self.s_max
        masks = self.get_masks(task_id, s)
        h = self.layer1(x) * masks[0]
        h = self.layer2(h) * masks[1]
        return self.heads[task_id](h)

    def hat_regularization(self, task_id: int,
                            current_masks: List[torch.Tensor],
                            c: float = 0.75) -> torch.Tensor:
        if task_id == 0:
            return torch.zeros(1, device=current_masks[0].device).squeeze()
        device = current_masks[0].device
        cum_masks = self.get_cumulative_mask(device=device)
        reg = torch.zeros(1, device=device)
        total = 0
        for cm, tm in zip(cum_masks, current_masks):
            reg = reg + (cm * tm).sum()
            total += cm.numel()
        return c * reg / total

    def compensate_gradients(self, task_id: int, current_masks: List[torch.Tensor]):
        if task_id == 0:
            return
        device = self.embeddings[0].device
        cum_masks = self.get_cumulative_mask(device=device)
        with torch.no_grad():
            for emb, cm in zip(self.embeddings, cum_masks):
                if emb.grad is not None:
                    emb.grad[:task_id].zero_()
                    emb.grad[task_id] *= (1.0 - cm)

    def complete_task(self, task_id: int):
        with torch.no_grad():
            for emb in self.embeddings:
                emb[task_id].clamp_(-1.0, 1.0)
        self._trained_tasks = task_id + 1
        pct = self._capacity_used_pct()
        print(f"[HAT] Task {task_id} complete. Capacity used: {pct:.1f}%")

    def _capacity_used_pct(self) -> float:
        if self._trained_tasks == 0:
            return 0.0
        cum = self.get_cumulative_mask()
        total = sum(c.numel() for c in cum)
        used = sum((c > 0.5).sum().item() for c in cum)
        return 100.0 * used / total

    def get_task_parameters(self, task_id: int):
        return list(self.parameters())
