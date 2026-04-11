"""
HAT — Hard Attention to the Task
=================================
Paper: "Overcoming Catastrophic Forgetting with Hard Attention to the Task"
       Serra, Suris, Miron, Karatzoglou — ICML 2018
https://arxiv.org/abs/1801.01423

Core idea:
  Each task t and each layer l has a learnable embedding vector a_t_l.
  At forward pass, a soft binary mask is computed:

      m_t_l = sigmoid(s · a_t_l)

  where s is a temperature that increases during training. At high s, the
  mask becomes approximately binary: units with a_t_l > 0 → mask ≈ 1,
  units with a_t_l < 0 → mask ≈ 0.

  The activations at each layer are element-wise multiplied by the mask,
  so units with mask ≈ 0 are effectively "off" for task t.

  Crucially, units that are ON (mask ≈ 1) for ANY previous task are
  protected: their weights receive zero gradient during new task training.
  This is enforced via a "gradient compensation" step after backward().

  Regularization: new task t is penalized for activating units that previous
  tasks already use:

      R = (1/n) * Σ_l Σ_i max_{t'<t}[m_{t',l,i}] · m_{t,l,i}

  This pushes new tasks to find unused capacity rather than overwrite existing.

Why this is better than PackNet:
  PackNet: hard binary masks chosen by magnitude pruning → greedy, suboptimal
  HAT: soft masks learned end-to-end → gradient finds the best capacity allocation

Why this is harder than EWC:
  HAT explicitly protects individual neurons rather than softly penalizing weights.
  No λ hyperparameter to balance old vs. new — the mask geometry does the work.

Known limitation: Still operates within fixed network capacity (same as PackNet).
As task count grows, the regularization forces new tasks into increasingly small
regions of the network, eventually limiting learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─── CIFAR HAT Model ─────────────────────────────────────────────────────────

class HATModel(nn.Module):
    """
    HAT with a 3-block CNN backbone for CIFAR images (3 × 32 × 32).

    Masks are applied per-channel after each conv block and per-unit after the FC.
    This matches the original paper's approach: one mask value per "neuron",
    where neurons in conv layers are channels (broadcast over spatial dims).

    Architecture:
      conv1 (3→64) → mask_1 (64d) → conv2 (64→128) → mask_2 (128d)
      → conv3 (128→256) → mask_3 (256d) → fc (→512) → mask_4 (512d) → Head[t]
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int = 2,
                 in_channels: int = 3, input_size: int = 32,
                 s_max: float = 400.0):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.s_max = s_max

        pooled = input_size // 8   # 3 maxpools of stride 2

        # ── Backbone (split for mask injection at each block) ──────────────
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

        # ── Per-task output heads ──────────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task)
            for _ in range(num_tasks)
        ])

        # ── HAT mask embeddings: shape (num_tasks, layer_output_dim) ──────
        # One ParameterList entry per layer. Each is (T, D_l).
        # Initialized near zero → mask ≈ 0.5 at low temperature (no prior preference).
        self.mask_dims = [64, 128, 256, 512]
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(num_tasks, d))
            for d in self.mask_dims
        ])

        # How many tasks have been fully trained (used for cumulative mask)
        self._trained_tasks = 0

    # ── Mask utilities ────────────────────────────────────────────────────────

    def get_masks(self, task_id: int, s: float) -> List[torch.Tensor]:
        """Compute soft masks at temperature s for task_id."""
        return [
            torch.sigmoid(s * self.embeddings[l][task_id])
            for l in range(len(self.mask_dims))
        ]

    @torch.no_grad()
    def get_cumulative_mask(self, device=None) -> List[torch.Tensor]:
        """
        Max mask over all trained tasks at near-binary temperature.
        Represents the "occupied" capacity of the network so far.
        """
        if self._trained_tasks == 0:
            return [torch.zeros(d) for d in self.mask_dims]

        s_test = self.s_max
        cum = []
        for l in range(len(self.mask_dims)):
            # Stack all trained tasks' masks, take element-wise max
            task_masks = torch.stack([
                torch.sigmoid(s_test * self.embeddings[l][t])
                for t in range(self._trained_tasks)
            ])  # (trained_tasks, D_l)
            cum.append(task_masks.max(0).values)

        if device is not None:
            cum = [c.to(device) for c in cum]
        return cum

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, task_id: int,
                s: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass with mask at temperature s.
        At test time (s=None), uses s_max → near-binary masks.
        """
        if s is None:
            s = self.s_max

        masks = self.get_masks(task_id, s)

        h = self.conv1(x)
        h = h * masks[0].view(1, -1, 1, 1)   # (B, 64, H, W) * (1, 64, 1, 1)

        h = self.conv2(h)
        h = h * masks[1].view(1, -1, 1, 1)

        h = self.conv3(h)
        h = h * masks[2].view(1, -1, 1, 1)

        h = self.fc_block(h)
        h = h * masks[3]                       # (B, 512) * (512,)

        return self.heads[task_id](h)

    # ── HAT loss ──────────────────────────────────────────────────────────────

    def hat_regularization(self, task_id: int,
                            current_masks: List[torch.Tensor],
                            c: float = 0.75) -> torch.Tensor:
        """
        Regularization term discouraging new task from reusing occupied capacity.

        R = c * (1/n) * Σ_l Σ_i Ω_{l,i} * m_{t,l,i}

        where Ω_{l,i} = max_{t'<t} m_{t',l,i}  (cumulative binary mask)
        """
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

    # ── Gradient compensation ─────────────────────────────────────────────────

    def compensate_gradients(self, task_id: int, current_masks: List[torch.Tensor]):
        """
        After backward(): scale down embedding gradients proportionally to
        how much a previous task has already claimed that unit.

        grad[t][i] *= (1 - Ω_i) where Ω_i = cumulative previous task mask

        This prevents the optimizer from pushing a_t_l for units that are
        fully claimed by previous tasks (Ω=1 → gradient zeroed).
        """
        if task_id == 0:
            return

        device = self.embeddings[0].device
        cum_masks = self.get_cumulative_mask(device=device)

        with torch.no_grad():
            for l, (emb, cm) in enumerate(zip(self.embeddings, cum_masks)):
                if emb.grad is not None:
                    # Zero gradient for previous tasks' embeddings (don't change them)
                    emb.grad[:task_id].zero_()
                    # Scale current task's embedding gradient by (1 - cumulative mask)
                    emb.grad[task_id] *= (1.0 - cm)

    # ── Post-task operations ──────────────────────────────────────────────────

    def complete_task(self, task_id: int):
        """
        Called after finishing training on task_id.
        Clips embeddings to stabilize masks at binary values.
        """
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


# ─── MNIST HAT Model ─────────────────────────────────────────────────────────

class HATMNISTModel(nn.Module):
    """
    HAT with 2-layer MLP backbone for Permuted MNIST.
    Mask applied after each hidden layer (per unit).
    """

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
