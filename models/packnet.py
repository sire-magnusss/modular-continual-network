"""
PackNet
--------
Paper: "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"
       — Mallya & Lazebnik, 2018
https://arxiv.org/abs/1711.05769

Core idea:
  Instead of penalizing weight changes (like EWC), PackNet physically prevents
  changes to weights that belong to past tasks by using hard binary masks.

  For each task:
    1. Train normally (all free weights)
    2. Prune: zero out the X% least-important (smallest magnitude) free weights
    3. Retrain: fine-tune only the pruned (freed) weights with frozen backbone
    4. Freeze: lock the surviving weights — they now "belong" to this task forever

  For task t+1, only the weights that were pruned/freed in step 2 are available.
  This allocates network capacity across tasks.

Why this is interesting:
  No forgetting by construction — frozen weights can't change.
  But capacity is finite: eventually you run out of free weights.
  This makes it a useful upper bound on "can we prevent forgetting with hard constraints?"

Known failure mode:
  After many tasks, the remaining free capacity is too small to learn new tasks well.
  This is the hard version of the plasticity problem — not a soft penalty, but a wall.

Implementation note:
  We work on the shared backbone only (features + fc), not the task heads.
  Task heads are always task-specific and never frozen across tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np


class PackNetModel(nn.Module):
    """
    PackNet wrapper for any backbone model.

    Usage:
        model = PackNetModel(backbone, prune_fraction=0.5)

        # Phase 1: train normally with model.forward(x, task_id)
        # Phase 2: prune and freeze
        model.prune_and_freeze(task_id)
        # Phase 3: retrain only free weights (optimizer should only get free params)
        free_params = model.get_free_parameters()
    """

    def __init__(self, backbone: nn.Module, prune_fraction: float = 0.5):
        """
        Args:
            backbone: CIFARBackbone or MNISTBackbone
            prune_fraction: fraction of currently-free weights to prune per task.
                            0.5 means half the remaining free weights get zeroed
                            and the other half get assigned to this task.
        """
        super().__init__()
        self.backbone = backbone
        self.prune_fraction = prune_fraction

        # Binary masks per parameter: True = frozen (belongs to a past task), False = free
        self._frozen_masks: Dict[str, torch.Tensor] = {}
        # Which task each weight belongs to (-1 = free/unassigned)
        self._task_masks: Dict[str, torch.Tensor] = {}
        # Snapshot of each parameter taken right after prune_and_freeze —
        # used to restore frozen weights after each optimizer step (prevents Adam momentum drift)
        self._frozen_snapshots: Dict[str, torch.Tensor] = {}

        self._initialize_masks()

    def _initialize_masks(self):
        """All weights start free (mask = 0)."""
        for name, param in self.backbone.named_parameters():
            if "heads" in name:
                continue  # task heads are not masked
            self._frozen_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
            self._task_masks[name] = torch.full_like(param.data, -1, dtype=torch.long)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.backbone(x, task_id)

    def apply_masks(self):
        """
        Restore frozen weights to their post-freeze snapshot values.
        Called after each optimizer.step() to undo any Adam momentum drift
        on frozen parameters (even with zero gradients, Adam can nudge weights).
        """
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in self._frozen_masks and name in self._frozen_snapshots:
                    mask = self._frozen_masks[name].to(param.device)
                    if mask.any():
                        snap = self._frozen_snapshots[name].to(param.device)
                        param.data[mask] = snap[mask]

    def prune_and_freeze(self, task_id: int):
        """
        After training phase 1 for task_id:
          1. Among free weights, zero the bottom prune_fraction by magnitude.
          2. Freeze the top (1 - prune_fraction) free weights → they belong to task_id.

        Returns the number of weights frozen for this task.
        """
        total_frozen = 0

        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name not in self._frozen_masks:
                    continue

                frozen_mask = self._frozen_masks[name]
                free_mask = ~frozen_mask  # True where weights are free

                free_indices = free_mask.nonzero(as_tuple=False)
                if free_indices.numel() == 0:
                    continue

                # Get magnitudes of free weights
                free_weights = param.data[free_mask]
                magnitudes = free_weights.abs()

                n_free = magnitudes.numel()
                n_to_freeze = max(1, int(n_free * (1 - self.prune_fraction)))

                # Find threshold: top n_to_freeze by magnitude get frozen
                if n_free > 0:
                    sorted_mags, sorted_idx = magnitudes.sort(descending=True)
                    freeze_threshold_idx = min(n_to_freeze, n_free - 1)
                    threshold = sorted_mags[freeze_threshold_idx].item()

                    # Build a mask of free weights to freeze
                    freeze_local = magnitudes >= threshold  # local index into free weights

                    # Map back to full parameter space
                    free_flat_indices = free_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
                    freeze_global_indices = free_flat_indices[freeze_local.view(-1)]

                    # Update frozen mask
                    flat_frozen = self._frozen_masks[name].view(-1)
                    flat_frozen[freeze_global_indices] = True
                    self._frozen_masks[name] = flat_frozen.view(param.shape)

                    # Record task ownership
                    flat_task = self._task_masks[name].view(-1)
                    flat_task[freeze_global_indices] = task_id
                    self._task_masks[name] = flat_task.view(param.shape)

                    # Zero out the pruned (remaining free) weights
                    param.data[~self._frozen_masks[name] & free_mask] = 0.0

                    total_frozen += freeze_global_indices.numel()

        # Save a snapshot of all parameters now — frozen weights will be restored
        # to these values after every optimizer step during subsequent task training
        for name, param in self.backbone.named_parameters():
            if name in self._frozen_masks:
                self._frozen_snapshots[name] = param.data.clone().cpu()

        pct = 100 * total_frozen / self._count_total_shared_params()
        free_pct = 100 * self._count_free_params() / self._count_total_shared_params()
        print(f"[PackNet] Task {task_id} frozen {total_frozen:,} weights ({pct:.1f}%). "
              f"Free capacity remaining: {free_pct:.1f}%")
        return total_frozen

    def get_free_parameters(self):
        """
        Generator yielding (name, param) for parameters that still have free weights.
        Pass these to the optimizer during retraining phase.
        """
        for name, param in self.backbone.named_parameters():
            if "heads" in name:
                yield name, param
            elif name in self._frozen_masks:
                if (~self._frozen_masks[name]).any():
                    yield name, param

    def freeze_gradients(self):
        """
        Hook to zero out gradients on frozen weights after backward pass.
        Call this after loss.backward() and before optimizer.step().
        """
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in self._frozen_masks and param.grad is not None:
                    param.grad[self._frozen_masks[name]] = 0.0

    def _count_total_shared_params(self) -> int:
        total = 0
        for name, param in self.backbone.named_parameters():
            if name in self._frozen_masks:
                total += param.numel()
        return max(total, 1)

    def _count_free_params(self) -> int:
        free = 0
        for name in self._frozen_masks:
            free += (~self._frozen_masks[name]).sum().item()
        return free

    @property
    def num_tasks(self):
        return self.backbone.num_tasks
