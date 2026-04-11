"""
MCN Ablation Variants
======================
Three stripped-down versions of MCN used to prove each component earns its place.

  mcn_no_router  — removes the attention router; just concatenates base + task features
  mcn_no_gate    — removes the sigmoid gate from the task module (always-on contribution)
  mcn_base_only  — no task module at all; frozen base encoder + task head only

Running all four (mcn + these three) gives you the ablation table:
  Component removed     | Avg Acc | Forgetting | Conclusion
  ──────────────────────────────────────────────────────────
  Nothing (full MCN)    |  86.5%  |    0.1%    | best
  No Router             |   ???   |    ???     | does attention help?
  No Gate               |   ???   |    ???     | does stable init help?
  Base Only             |   ???   |    ???     | does the task module help?
"""

import torch
import torch.nn as nn
from models.mcn import MCN, TaskModule


# ─── MCN without Router ───────────────────────────────────────────────────────

class MCNNoRouter(MCN):
    """
    Removes the attention router. Instead of learned blending, just
    concatenates base_feat and task_feat and projects to base_dim.

    Tests: does the router's attention mechanism actually help,
           or is simple concatenation enough?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace routers with simple linear projections
        concat_dim = self.base_dim + 256  # base_dim + task_dim
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(concat_dim, self.base_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        base_feat = self.base_high(self.base_low(x))
        task_feat = self.task_modules[task_id](x)
        # Simple concat + project instead of attention
        blended = self.routers[task_id](torch.cat([base_feat, task_feat], dim=1))
        return self.heads[task_id](blended)

    def get_task_parameters(self, task_id: int):
        if not self._base_frozen:
            return list(self.parameters())
        params = []
        params.extend(self.task_modules[task_id].parameters())
        params.extend(self.routers[task_id].parameters())
        params.extend(self.heads[task_id].parameters())
        return params


# ─── MCN without Gate ─────────────────────────────────────────────────────────

class MCNNoGate(MCN):
    """
    Removes the sigmoid gate from TaskModule.
    The task module always contributes at full strength from the start.

    Tests: does the gated initialization (module starts contributing near 0,
           gradually opens) help training stability vs full contribution immediately?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace task modules with ungated versions
        self.task_modules = nn.ModuleList([
            _UngatedTaskModule(
                in_channels=self.in_channels,
                out_dim=256,
                input_size=self.input_size
            )
            for _ in range(self.num_tasks)
        ])


class _UngatedTaskModule(nn.Module):
    """TaskModule with the sigmoid gate removed — always contributes at full strength."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256,
                 input_size: int = 32):
        super().__init__()
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
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv_blocks(x))  # no gate — full contribution always


# ─── MCN Base Only ────────────────────────────────────────────────────────────

class MCNBaseOnly(nn.Module):
    """
    No task module. No router. Just:
      frozen base encoder → task-specific head

    This is the lower bound: proves that task modules + router are
    responsible for the accuracy gains over a plain frozen backbone.

    If this performs poorly on new tasks → task modules are necessary.
    If this performs well → the base encoder alone is sufficient (no need for MCN).
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int,
                 base_dim: int = 512, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.base_dim = base_dim
        self.in_channels = in_channels
        self.input_size = input_size

        pooled = input_size // 8
        self.base_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256 * pooled * pooled, base_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleList([
            nn.Linear(base_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

        self._base_frozen = False

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[task_id](self.base_encoder(x))

    def freeze_base_encoder(self):
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        self._base_frozen = True
        print("[MCN-BaseOnly] Base encoder frozen.")

    def get_task_parameters(self, task_id: int):
        if not self._base_frozen:
            return list(self.parameters())
        return list(self.heads[task_id].parameters())
