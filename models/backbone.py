"""
Shared CNN backbones used by all methods.

CIFARBackbone  — for Split-CIFAR-10 (3x32x32 input, 2 classes per task)
MNISTBackbone  — for Permuted MNIST  (1x28x28 input, 10 classes per task)

Architecture philosophy:
  Small enough to train fast on M4 MacBook (~30s/epoch).
  Deep enough to be non-trivial and show real forgetting dynamics.
  Uses task-specific heads so each task gets its own output layer —
  this is the "multi-head" setup standard in continual learning research.
"""

import torch
import torch.nn as nn


# ─── CIFAR Backbone ──────────────────────────────────────────────────────────

class CIFARBackbone(nn.Module):
    """
    3-block conv network for CIFAR-32x32 images.

    Block structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool

    Feature extractor outputs 512-dim embedding.
    Each task gets its own linear head (num_classes_per_task outputs).
    """

    def __init__(self, num_tasks: int, num_classes_per_task: int = 2):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # Shared feature extractor
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32 -> 16x16

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16 -> 8x8

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8x8 -> 4x4
        )

        # 256 * 4 * 4 = 4096 -> 512
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # One head per task
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return self.heads[task_id](x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim embedding before the head (useful for analysis)."""
        x = self.features(x)
        return self.fc(x)

    def shared_parameters(self):
        """Generator over backbone params only (not heads)."""
        yield from self.features.parameters()
        yield from self.fc.parameters()


# ─── MNIST Backbone ───────────────────────────────────────────────────────────

class MNISTBackbone(nn.Module):
    """
    Fully-connected network for Permuted MNIST (flattened 784-dim input).
    Multi-layer perceptron with shared layers + per-task heads.

    Note: For permuted MNIST we use MLP not CNN, because random pixel
    permutations break spatial structure — a CNN would be unfair.
    """

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
