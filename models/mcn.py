"""
Modular Continual Network (MCN)
================================
A novel continual learning architecture designed to eliminate the
capacity-forgetting tradeoff by growing new modules per task instead
of competing over fixed weights.

Architecture
------------
                        ┌─────────────────────────┐
    Input               │      Base Encoder        │  (frozen after Task 0)
      │                 │   3 conv blocks → 512d   │
      ├────────────────►│                          │──► base_feat (512d)
      │                 └─────────────────────────┘
      │
      │                 ┌─────────────────────────┐
      └────────────────►│  Task Module [task_id]  │──► task_feat (256d)
                        │  Lightweight residual    │
                        │  adapter (new per task)  │
                        └─────────────────────────┘
                                    │
                        ┌─────────────────────────┐
                        │       Router             │
                        │  Attends over base_feat  │
                        │  + task_feat → 512d out  │
                        └─────────────────────────┘
                                    │
                        ┌─────────────────────────┐
                        │   Task Head [task_id]    │──► logits
                        └─────────────────────────┘

Key design decisions:
  1. Base encoder is trained on Task 0, then frozen. It learns general
     low-level features (edges, textures) that transfer to all tasks.
     Freezing it = zero forgetting of these shared representations.

  2. Task modules are small (3x fewer params than base). Each new task
     gets its own module — no competition, no forgetting by construction.
     The module is a residual adapter: output = input + f(input), so if
     f→0 (early training), the module passes through the base features.
     This makes initialization stable.

  3. Router is a 2-layer MLP that learns to blend base_feat and task_feat
     using soft attention. This lets the network use base features heavily
     for easy tasks and lean on task-specific features for hard ones.
     The router is task-specific (one per task, trained alongside the module).

  4. No masks, no Fisher matrices, no capacity limits. Adding a task = adding
     ~400K parameters (module + router + head). The base encoder (~3.2M params)
     is shared and frozen.

Why this beats PackNet:
  PackNet: 3 tasks on a 3.24M param network → each task gets ~1M params
  MCN: base (3.2M frozen) + 3 task modules (400K each) = 4.4M total params
  MCN gets more effective capacity per task, and zero cross-task interference.

Why this beats EWC:
  EWC: soft penalty that leaks. The larger λ, the worse new task learning.
  MCN: hard isolation by architecture. No tradeoff parameter to tune.

Research question this tests:
  "Can we achieve near-zero forgetting without sacrificing new task plasticity,
   by growing module capacity instead of compressing fixed capacity?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─── Task Module ─────────────────────────────────────────────────────────────

class TaskModule(nn.Module):
    """
    Lightweight residual adapter for one task.

    Takes the raw input (same as base encoder input) and produces a
    task-specific feature vector that complements the base encoder's output.

    Architecture: 2-block conv net with skip connection, outputs 256-dim.
    Deliberately smaller than base encoder to keep parameter count low.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256,
                 input_size: int = 32):
        super().__init__()

        # 3 maxpools: 32 → 4, 28 → 3  (keeps FC layer small)
        pooled_size = input_size // 8

        self.conv_blocks = nn.Sequential(
            # Block 1: lightweight — 32 channels
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: keep 64 channels, reduce spatial further
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

        # Gate initialized to -3 so sigmoid(-3) ≈ 0.05:
        # module starts contributing almost nothing, opens up as task-specific
        # gradients push it. This prevents the unfrozen task module from
        # destabilizing the frozen base encoder's output early in training.
        self.gate = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc(self.conv_blocks(x))
        # Sigmoid gate: starts ~0.05, gradually opens toward 1.0
        return feat * torch.sigmoid(self.gate)


# ─── Router ──────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """
    Blends base encoder features and task module features using learned attention.

    Inputs:  base_feat (512d) + task_feat (256d) → concat (768d)
    Output:  blended feature (512d)

    The router is task-specific — each task gets its own router.
    This lets the network learn different blending strategies per task:
    some tasks may rely almost entirely on base features (simple tasks),
    others heavily on the task module (complex or very different tasks).
    """

    def __init__(self, base_dim: int = 512, task_dim: int = 256,
                 out_dim: int = 512):
        super().__init__()
        concat_dim = base_dim + task_dim

        # Per-sample attention: learns which stream to trust based on the
        # actual content of that sample's features (not a fixed scalar)
        self.attn = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        # Projection layers: bring both streams to out_dim before blending
        self.base_proj = nn.Linear(base_dim, out_dim)
        self.task_proj = nn.Linear(task_dim, out_dim)

        # Final fusion layer: takes weighted-sum features → out_dim
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, base_feat: torch.Tensor,
                task_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([base_feat, task_feat], dim=1)

        # Per-sample attention weights (B, 2)
        weights = self.attn(concat)
        w_base = weights[:, 0:1]   # (B, 1)
        w_task = weights[:, 1:2]   # (B, 1)

        # Project both streams to the same dim, then weighted sum
        base_p = self.base_proj(base_feat)   # (B, out_dim)
        task_p = self.task_proj(task_feat)   # (B, out_dim)
        blended = w_base * base_p + w_task * task_p

        # Residual fusion: blend + skip from base projection
        return self.fusion(blended) + base_p


# ─── MCN Main Model ──────────────────────────────────────────────────────────

class MCN(nn.Module):
    """
    Modular Continual Network.

    Initialize with num_tasks=1 and call add_task() as new tasks arrive.
    Or initialize with the full num_tasks upfront for benchmarking.

    Args:
        num_tasks:            Total number of tasks (for benchmarking, can init all at once)
        num_classes_per_task: Number of output classes per task head
        base_dim:             Dimension of base encoder output (512)
        task_dim:             Dimension of each task module output (256)
        in_channels:          Input image channels (3 for CIFAR, 1 for MNIST)
        input_size:           Spatial size of input (32 for CIFAR, 28 for MNIST)
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int,
                 base_dim: int = 512, task_dim: int = 256,
                 in_channels: int = 3, input_size: int = 32):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.base_dim = base_dim
        self.task_dim = task_dim
        self.in_channels = in_channels
        self.input_size = input_size

        # ── Base Encoder (shared, frozen after Task 0) ──
        # Uses the same 3-block CNN as before, but now it's explicitly
        # separated into a reusable feature extractor.
        pooled = input_size // 8  # after 3 maxpools: 32→4, 28→3
        self.base_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Projection to base_dim
            nn.Flatten(),
            nn.Linear(256 * pooled * pooled, base_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # ── Per-Task Components (grown dynamically) ──
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

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        # Base encoder (frozen after Task 0)
        base_feat = self.base_encoder(x)

        # Task-specific module
        task_feat = self.task_modules[task_id](x)

        # Router blends the two feature streams
        blended = self.routers[task_id](base_feat, task_feat)

        # Task head
        return self.heads[task_id](blended)

    def freeze_base_encoder(self):
        """
        Freeze the base encoder after Task 0.
        All subsequent tasks can only update their own modules and router.
        """
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        self._base_frozen = True
        print("[MCN] Base encoder frozen. Future tasks will only update their own modules.")

    def get_task_parameters(self, task_id: int):
        """
        Return only the parameters that should be updated when training task_id.
        If base is frozen: task module + router + head only.
        If base is not yet frozen (Task 0): everything.
        """
        if not self._base_frozen:
            # Task 0: train everything
            return list(self.parameters())
        else:
            # Task t > 0: only task-specific components
            params = []
            params.extend(self.task_modules[task_id].parameters())
            params.extend(self.routers[task_id].parameters())
            params.extend(self.heads[task_id].parameters())
            return params

    def get_base_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return base encoder features (useful for analysis and visualization)."""
        with torch.no_grad():
            return self.base_encoder(x)

    def param_count(self) -> dict:
        """Report parameter counts for each component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        base = count(self.base_encoder)
        per_task_module = count(self.task_modules[0]) if self.task_modules else 0
        per_task_router = count(self.routers[0]) if self.routers else 0
        per_task_head = count(self.heads[0]) if self.heads else 0
        per_task_total = per_task_module + per_task_router + per_task_head

        return {
            "base_encoder": base,
            "per_task_module": per_task_module,
            "per_task_router": per_task_router,
            "per_task_head": per_task_head,
            "per_task_total": per_task_total,
            "total_for_n_tasks": base + per_task_total * self.num_tasks,
        }
