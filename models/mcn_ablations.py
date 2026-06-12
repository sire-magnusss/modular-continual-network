"""Ablation variants for MCN."""

import torch
import torch.nn as nn
from models.mcn import MCN, TaskModule


# MCN without router

class MCNNoRouter(MCN):
    """Replace the attention router with concat + linear projection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        concat_dim = self.base_dim + 256
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


# MCN without gate

class MCNNoGate(MCN):
    """Use ungated task modules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.task_modules = nn.ModuleList([
            _UngatedTaskModule(
                in_channels=self.in_channels,
                out_dim=256,
                input_size=self.input_size
            )
            for _ in range(self.num_tasks)
        ])


class _UngatedTaskModule(nn.Module):
    """TaskModule without the sigmoid gate."""

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
        return self.fc(self.conv_blocks(x))


# MCN base only

class MCNBaseOnly(nn.Module):
    """Frozen base encoder with task-specific heads only."""

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
