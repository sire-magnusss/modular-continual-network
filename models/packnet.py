"""PackNet-style pruning baseline."""

import torch
import torch.nn as nn
from typing import Dict


class PackNetModel(nn.Module):
    """Wrap a backbone with pruning and frozen-weight masks."""

    def __init__(self, backbone: nn.Module, prune_fraction: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.prune_fraction = prune_fraction

        self._frozen_masks: Dict[str, torch.Tensor] = {}
        self._task_masks: Dict[str, torch.Tensor] = {}
        self._frozen_snapshots: Dict[str, torch.Tensor] = {}

        self._initialize_masks()

    def _initialize_masks(self):
        """Initialize all shared weights as free."""
        for name, param in self.backbone.named_parameters():
            if "heads" in name:
                continue
            self._frozen_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
            self._task_masks[name] = torch.full_like(param.data, -1, dtype=torch.long)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.backbone(x, task_id)

    def apply_masks(self):
        """Restore frozen weights after an optimizer step."""
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in self._frozen_masks and name in self._frozen_snapshots:
                    mask = self._frozen_masks[name].to(param.device)
                    if mask.any():
                        snap = self._frozen_snapshots[name].to(param.device)
                        param.data[mask] = snap[mask]

    def prune_and_freeze(self, task_id: int):
        """Prune low-magnitude free weights and freeze the remaining ones."""
        total_frozen = 0

        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name not in self._frozen_masks:
                    continue

                frozen_mask = self._frozen_masks[name]
                free_mask = ~frozen_mask

                free_indices = free_mask.nonzero(as_tuple=False)
                if free_indices.numel() == 0:
                    continue

                free_weights = param.data[free_mask]
                magnitudes = free_weights.abs()

                n_free = magnitudes.numel()
                n_to_freeze = max(1, int(n_free * (1 - self.prune_fraction)))

                if n_free > 0:
                    sorted_mags, sorted_idx = magnitudes.sort(descending=True)
                    freeze_threshold_idx = min(n_to_freeze, n_free - 1)
                    threshold = sorted_mags[freeze_threshold_idx].item()

                    freeze_local = magnitudes >= threshold

                    free_flat_indices = free_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
                    freeze_global_indices = free_flat_indices[freeze_local.view(-1)]

                    flat_frozen = self._frozen_masks[name].view(-1)
                    flat_frozen[freeze_global_indices] = True
                    self._frozen_masks[name] = flat_frozen.view(param.shape)

                    flat_task = self._task_masks[name].view(-1)
                    flat_task[freeze_global_indices] = task_id
                    self._task_masks[name] = flat_task.view(param.shape)

                    param.data[~self._frozen_masks[name] & free_mask] = 0.0

                    total_frozen += freeze_global_indices.numel()

        for name, param in self.backbone.named_parameters():
            if name in self._frozen_masks:
                self._frozen_snapshots[name] = param.data.clone().cpu()

        pct = 100 * total_frozen / self._count_total_shared_params()
        free_pct = 100 * self._count_free_params() / self._count_total_shared_params()
        print(f"[PackNet] Task {task_id} frozen {total_frozen:,} weights ({pct:.1f}%). "
              f"Free capacity remaining: {free_pct:.1f}%")
        return total_frozen

    def get_free_parameters(self):
        """Yield parameters that still contain trainable weights."""
        for name, param in self.backbone.named_parameters():
            if "heads" in name:
                yield name, param
            elif name in self._frozen_masks:
                if (~self._frozen_masks[name]).any():
                    yield name, param

    def freeze_gradients(self):
        """Zero gradients for frozen weights."""
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
