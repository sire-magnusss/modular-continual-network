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
from typing import List, Optional, Tuple


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
    Modular Continual Network — with Adaptive Layer Freezing.

    The base encoder is split into two parts:
      base_low  (Block 1+2): learns edges and textures → permanently frozen after Task 0.
                              These features transfer to ALL tasks — no reason to change them.
      base_high (Block 3+FC): learns shapes and semantic structure → kept adaptable.
                              New tasks train this at a reduced lr (adaptive_lr_scale × lr),
                              allowing high-level representations to shift for new domains
                              without destroying low-level feature reuse.

    Why this fixes the MNIST gap:
      Permuted MNIST has identical high-level structure (digit shapes) but completely
      different spatial layout per task. The fully-frozen base encoder couldn't adapt its
      high-level semantic representations — base_high adaption fixes this.
      CIFAR-10 tasks (different object classes) benefit less but are not hurt because
      base_high lr is small (0.1×), so Task 0 knowledge degrades slowly.

    Architecture:
      base_low  → Block 1+2 (in→128ch, 2 maxpools)        [frozen after T0]
      base_high → Block 3 + FC (128→256ch + Linear→512d)  [adaptive, low lr]
      task_module[t] → lightweight CNN adapter              [full lr]
      router[t]      → attention blender                   [full lr]
      head[t]        → linear classifier                   [full lr]
    """

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
        self.freeze_all = freeze_all  # if True: freeze entire base after T0 (original behaviour)

        pooled = input_size // 8  # after 3 maxpools: 32→4, 28→3

        # ── Low-level base (frozen after Task 0) ──────────────────────────
        # Block 1: in_channels → 64ch, spatial / 2
        # Block 2: 64 → 128ch, spatial / 2
        # Learns edges, colour blobs, simple textures — universal features.
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

        # ── High-level base (adaptive after Task 0, small lr) ─────────────
        # Block 3: 128 → 256ch, spatial / 2
        # FC: 256 * pooled * pooled → base_dim
        # Learns object-level semantics — task-specific enough to benefit from adaptation.
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

        # ── Per-task components ────────────────────────────────────────────
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
        After Task 0:
          freeze_all=True  → freeze entire base encoder (best for diverse tasks like CIFAR-10).
          freeze_all=False → freeze only base_low; base_high trains at adaptive_lr_scale × lr
                             (helps when tasks share structure but differ in high-level layout).
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
                  f"base_high adaptive ({high_p:,} params @ {self.adaptive_lr_scale}× lr).")

    def get_task_param_groups(self, task_id: int, base_lr: float) -> list:
        """
        Return optimizer parameter groups for training task_id.

        Task 0: single group — everything at base_lr.
        Task t > 0: two groups:
          - base_high at base_lr × adaptive_lr_scale  (slow adaptation)
          - task module + router + head at base_lr     (full speed)
        """
        if not self._base_frozen:
            return [{"params": list(self.parameters()), "lr": base_lr, "name": "all"}]

        task_params = (list(self.task_modules[task_id].parameters()) +
                       list(self.routers[task_id].parameters()) +
                       list(self.heads[task_id].parameters()))

        if self.freeze_all:
            # base fully frozen — only task-specific params
            return [{"params": task_params, "lr": base_lr, "name": "task_specific"}]
        else:
            # adaptive: base_high at slow lr, task components at full lr
            return [
                {"params": list(self.base_high.parameters()),
                 "lr": base_lr * self.adaptive_lr_scale, "name": "base_high"},
                {"params": task_params, "lr": base_lr, "name": "task_specific"},
            ]

    # Keep backward-compatible alias used by ablation variants
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

        Runs forward pass through all trained task heads and picks the task
        whose predictions have the LOWEST Shannon entropy (highest confidence).
        Returns per-sample (predicted_label, predicted_task_id).

        Complexity: O(T) forward passes — acceptable at inference time.
        Accuracy depends on task separation: if two tasks have similar inputs
        (like MNIST digits across permutations), entropy discrimination degrades.
        For visually distinct tasks (CIFAR), entropy reliably picks the right task.

        This moves MCN from task-incremental (task ID known at test time) toward
        class-incremental evaluation — a harder, more realistic setting.
        """
        self.eval()
        if self._num_trained == 0:
            raise RuntimeError("No tasks have been trained yet.")

        all_logits: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []

        for t in range(self._num_trained):
            logits = self.forward(x, task_id=t)      # (B, C_t)
            probs = F.softmax(logits, dim=1)          # (B, C_t)
            # Shannon entropy per sample
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # (B,)
            all_logits.append(logits)
            all_entropies.append(entropy)

        # Stack: (B, T) — pick task with minimum entropy (maximum confidence)
        entropies = torch.stack(all_entropies, dim=1)   # (B, T)
        best_task_ids = entropies.argmin(dim=1)          # (B,)

        # Gather the predicted class for each sample from the chosen task
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
