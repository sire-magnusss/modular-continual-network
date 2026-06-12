"""Elastic Weight Consolidation baseline."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


class EWCModel(nn.Module):
    """Wrap a backbone with an EWC penalty."""

    def __init__(self, backbone: nn.Module, ewc_lambda: float = 5000.0):
        super().__init__()
        self.backbone = backbone
        self.ewc_lambda = ewc_lambda

        self._consolidations: Dict[int, dict] = {}

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.backbone(x, task_id)

    def consolidate(self, task_id: int, train_loader: DataLoader,
                    device: torch.device, num_samples: int = 1000):
        """Store parameter snapshots and diagonal Fisher estimates."""
        self.backbone.eval()

        params_snapshot = {
            name: param.clone().detach()
            for name, param in self.backbone.named_parameters()
        }

        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.backbone.named_parameters()
        }

        samples_seen = 0
        for x, y in train_loader:
            if samples_seen >= num_samples:
                break
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            self.backbone.zero_grad()
            logits = self.backbone(x, task_id)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, y)
            loss.backward()

            for name, param in self.backbone.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2) * batch_size

            samples_seen += batch_size

        for name in fisher:
            fisher[name] /= min(samples_seen, num_samples)

        self._consolidations[task_id] = {
            "params": params_snapshot,
            "fisher": fisher,
        }

        self.backbone.train()
        print(f"[EWC] Consolidated task {task_id} "
              f"(lambda={self.ewc_lambda}, {samples_seen} samples)")

    def ewc_penalty(self) -> torch.Tensor:
        """Return the accumulated EWC penalty."""
        if not self._consolidations:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for task_id, consolidation in self._consolidations.items():
            for name, param in self.backbone.named_parameters():
                mean = consolidation["params"][name]
                fisher = consolidation["fisher"][name]
                penalty = penalty + (fisher * (param - mean).pow(2)).sum()

        return self.ewc_lambda / 2.0 * penalty

    @property
    def num_tasks(self):
        return self.backbone.num_tasks
