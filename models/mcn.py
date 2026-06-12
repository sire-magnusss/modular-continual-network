"""Modular Continual Network model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# Task module

class TaskModule(nn.Module):
    """Task-specific CNN adapter that outputs a feature vector."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256,
                 input_size: int = 32):
        super().__init__()

        # Three max-pools reduce 32x32 to 4x4 and 28x28 to 3x3.
        pooled_size = input_size // 8

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        flat_dim = 64 * pooled_size * pooled_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
        )

        # Start the adapter contribution small and let training open it.
        self.gate = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc(self.conv_blocks(x))
        return feat * torch.sigmoid(self.gate)


# Router

class Router(nn.Module):
    """Blend base and task-specific features with learned attention."""

    def __init__(self, base_dim: int = 512, task_dim: int = 256,
                 out_dim: int = 512):
        super().__init__()
        concat_dim = base_dim + task_dim

        self.attn = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        self.base_proj = nn.Linear(base_dim, out_dim)
        self.task_proj = nn.Linear(task_dim, out_dim)

        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, base_feat: torch.Tensor,
                task_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([base_feat, task_feat], dim=1)

        weights = self.attn(concat)
        w_base = weights[:, 0:1]
        w_task = weights[:, 1:2]

        base_p = self.base_proj(base_feat)
        task_p = self.task_proj(task_feat)
        blended = w_base * base_p + w_task * task_p

        return self.fusion(blended) + base_p


# MCN model

class MCN(nn.Module):
    """Base encoder plus task-specific adapters, routers, and heads."""

    def __init__(self, num_tasks: int, num_classes_per_task: int,
                 base_dim: int = 512, task_dim: int = 256,
                 in_channels: int = 3, input_size: int = 32,
                 adaptive_lr_scale: float = 0.1,
                 freeze_all: bool = False):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.base_dim = base_dim
        self.task_dim = task_dim
        self.in_channels = in_channels
        self.input_size = input_size
        self.adaptive_lr_scale = adaptive_lr_scale
        self.freeze_all = freeze_all

        pooled = input_size // 8

        # Low-level base, frozen after Task 0.
        self.base_low = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # High-level base. It is either frozen or trained with a reduced LR.
        self.base_high = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256 * pooled * pooled, base_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.task_modules = nn.ModuleList([
            TaskModule(in_channels, task_dim, input_size)
            for _ in range(num_tasks)
        ])

        self.routers = nn.ModuleList([
            Router(base_dim, task_dim, base_dim)
            for _ in range(num_tasks)
        ])

        self.heads = nn.ModuleList([
            nn.Linear(base_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

        self._base_frozen = False
        self._num_trained = 0   # updated by MCNTrainer after each task

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        base_feat = self.base_high(self.base_low(x))
        task_feat = self.task_modules[task_id](x)
        blended   = self.routers[task_id](base_feat, task_feat)
        return self.heads[task_id](blended)

    def freeze_base_encoder(self):
        """
        Freeze the base after Task 0.

        If freeze_all is false, base_high remains trainable at a reduced
        learning rate through get_task_param_groups().
        """
        for param in self.base_low.parameters():
            param.requires_grad = False
        if self.freeze_all:
            for param in self.base_high.parameters():
                param.requires_grad = False
        self._base_frozen = True
        low_p  = sum(p.numel() for p in self.base_low.parameters())
        high_p = sum(p.numel() for p in self.base_high.parameters())
        if self.freeze_all:
            print(f"[MCN] Full base frozen ({low_p+high_p:,} params). "
                  f"Task modules get full plasticity.")
        else:
            print(f"[MCN] base_low frozen ({low_p:,} params). "
                  f"base_high adaptive ({high_p:,} params @ {self.adaptive_lr_scale}x lr).")

    def get_task_param_groups(self, task_id: int, base_lr: float) -> list:
        """
        Return optimizer parameter groups for training task_id.

        Task 0: single group at base_lr.
        Task t > 0: two groups:
          - base_high at base_lr * adaptive_lr_scale
          - task module, router, and head at base_lr
        """
        if not self._base_frozen:
            return [{"params": list(self.parameters()), "lr": base_lr, "name": "all"}]

        task_params = (list(self.task_modules[task_id].parameters()) +
                       list(self.routers[task_id].parameters()) +
                       list(self.heads[task_id].parameters()))

        if self.freeze_all:
            return [{"params": task_params, "lr": base_lr, "name": "task_specific"}]
        else:
            return [
                {"params": list(self.base_high.parameters()),
                 "lr": base_lr * self.adaptive_lr_scale, "name": "base_high"},
                {"params": task_params, "lr": base_lr, "name": "task_specific"},
            ]

    # Backward-compatible alias used by ablation variants.
    def get_task_parameters(self, task_id: int):
        if not self._base_frozen:
            return list(self.parameters())
        return (list(self.base_high.parameters()) +
                list(self.task_modules[task_id].parameters()) +
                list(self.routers[task_id].parameters()) +
                list(self.heads[task_id].parameters()))

    @torch.no_grad()
    def predict_task_free(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Task-free inference: predict class label without knowing the task ID.

        Run all trained task heads and choose the lowest-entropy prediction.
        Returns per-sample (predicted_label, predicted_task_id).

        Complexity is O(T) forward passes.
        Accuracy depends on task separation: if two tasks have similar inputs
        (like MNIST digits across permutations), entropy discrimination degrades.
        For visually distinct tasks (CIFAR), entropy reliably picks the right task.

        This approximates task-free evaluation by selecting a task head from
        model confidence.
        """
        self.eval()
        if self._num_trained == 0:
            raise RuntimeError("No tasks have been trained yet.")

        all_logits: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []

        for t in range(self._num_trained):
            logits = self.forward(x, task_id=t)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
            all_logits.append(logits)
            all_entropies.append(entropy)

        entropies = torch.stack(all_entropies, dim=1)
        best_task_ids = entropies.argmin(dim=1)

        B = x.size(0)
        predicted_labels = torch.stack([
            all_logits[best_task_ids[i].item()][i].argmax()
            for i in range(B)
        ])

        return predicted_labels, best_task_ids

    def get_base_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base_high(self.base_low(x))

    def param_count(self) -> dict:
        def count(m): return sum(p.numel() for p in m.parameters())
        base_low   = count(self.base_low)
        base_high  = count(self.base_high)
        per_module = count(self.task_modules[0]) if self.task_modules else 0
        per_router = count(self.routers[0])       if self.routers else 0
        per_head   = count(self.heads[0])          if self.heads else 0
        per_total  = per_module + per_router + per_head
        return {
            "base_encoder":    base_low + base_high,
            "base_low":        base_low,
            "base_high":       base_high,
            "per_task_module": per_module,
            "per_task_router": per_router,
            "per_task_head":   per_head,
            "per_task_total":  per_total,
            "total_for_n_tasks": base_low + base_high + per_total * self.num_tasks,
        }
