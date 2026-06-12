"""Shared backbone models for continual-learning baselines."""

import torch
import torch.nn as nn


# CIFAR backbone

class CIFARBackbone(nn.Module):
    """CNN backbone with one output head per task."""

    def __init__(self, num_tasks: int, num_classes_per_task: int = 2):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return self.heads[task_id](x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding before the task head."""
        x = self.features(x)
        return self.fc(x)

    def shared_parameters(self):
        """Yield feature-extractor parameters, excluding task heads."""
        yield from self.features.parameters()
        yield from self.fc.parameters()


# MNIST backbone

class MNISTBackbone(nn.Module):
    """MLP backbone with one output head per task."""

    def __init__(self, num_tasks: int, num_classes_per_task: int = 10):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([
            nn.Linear(400, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.features(x)
        return self.heads[task_id](x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def shared_parameters(self):
        yield from self.features.parameters()
