"""
Elastic Weight Consolidation (EWC)
------------------------------------
Paper: "Overcoming catastrophic forgetting in neural networks" — Kirkpatrick et al., 2017
https://arxiv.org/abs/1612.00796

Core idea:
  After training on task t, compute the Fisher Information Matrix (FIM) for each
  parameter. The FIM approximates how important each parameter is to task t's
  performance. When training on task t+1, add a quadratic penalty that resists
  changes to important parameters:

    L_total = L_task(t+1) + λ/2 * Σ_i F_i * (θ_i - θ*_i)²

  where:
    F_i   = Fisher information for parameter i (diagonal approximation)
    θ*_i  = optimal parameter value after task t
    λ     = regularization strength (hyperparameter)

Why diagonal FIM?
  The full FIM is O(params²) — impractical. The diagonal approximation is
  the expected squared gradient, computed as the mean of squared gradients
  over the task's training data. This is cheap and works surprisingly well.

Known failure mode you'll observe:
  With many tasks, the FIM penalties accumulate and the network becomes
  increasingly rigid — eventually learning new tasks becomes hard too.
  This is called "the rigidity-plasticity dilemma."
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict


class EWCModel(nn.Module):
    """
    Wrapper that adds EWC regularization to any backbone model.

    Usage:
        model = EWCModel(backbone, ewc_lambda=5000)
        # After each task:
        model.consolidate(task_id, train_loader, device)
        # During training:
        loss = criterion(logits, labels) + model.ewc_penalty()
    """

    def __init__(self, backbone: nn.Module, ewc_lambda: float = 5000.0):
        super().__init__()
        self.backbone = backbone
        self.ewc_lambda = ewc_lambda

        # Stored per completed task: {task_id: {"params": ..., "fisher": ...}}
        self._consolidations: Dict[int, dict] = {}

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.backbone(x, task_id)

    def consolidate(self, task_id: int, train_loader: DataLoader,
                    device: torch.device, num_samples: int = 1000):
        """
        After finishing task_id, compute and store:
          - θ* (current parameters, the optimal point for this task)
          - F  (diagonal Fisher Information, estimated from training data)
        """
        self.backbone.eval()

        # Store a snapshot of current parameters
        params_snapshot = {
            name: param.clone().detach()
            for name, param in self.backbone.named_parameters()
        }

        # Estimate diagonal Fisher Information
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
            # Use log-softmax output: E[grad² of log p(y|x,θ)]
            log_probs = torch.log_softmax(logits, dim=1)
            # Sample from the model's distribution for Fisher estimation
            # (in practice, using true labels is also common and fine)
            loss = torch.nn.functional.nll_loss(log_probs, y)
            loss.backward()

            for name, param in self.backbone.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2) * batch_size

            samples_seen += batch_size

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= min(samples_seen, num_samples)

        self._consolidations[task_id] = {
            "params": params_snapshot,
            "fisher": fisher,
        }

        self.backbone.train()
        print(f"[EWC] Consolidated task {task_id} "
              f"(λ={self.ewc_lambda}, {samples_seen} samples)")

    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute the total EWC regularization penalty across all consolidated tasks.
        Call this and add to your task loss during training.
        """
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
