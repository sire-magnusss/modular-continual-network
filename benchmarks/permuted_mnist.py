"""Permuted MNIST benchmark."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class PermutedMNIST:
    """Build train/test loaders for fixed pixel permutations of MNIST."""

    num_classes_per_task = 10
    input_shape = (1, 28, 28)
    name = "PermutedMNIST"

    def __init__(self, num_tasks: int = 5, data_dir: str = "./data",
                 batch_size: int = 256, num_workers: int = 0, seed: int = 42):
        self.num_tasks = num_tasks
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.permutation_seed = seed

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self._train_full = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        self._test_full = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

        rng = np.random.RandomState(seed)
        n_pixels = 28 * 28
        self._permutations = [np.arange(n_pixels)]
        for _ in range(num_tasks - 1):
            self._permutations.append(rng.permutation(n_pixels))

    def get_train_loader(self, task_id: int) -> DataLoader:
        perm = self._permutations[task_id]
        ds = _PermutedDataset(self._train_full, perm)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def get_test_loader(self, task_id: int) -> DataLoader:
        perm = self._permutations[task_id]
        ds = _PermutedDataset(self._test_full, perm)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def task_description(self, task_id: int) -> str:
        if task_id == 0:
            return "Task 0: Original MNIST"
        return f"Task {task_id}: Permuted MNIST (seed variant {task_id})"

    def to_global_labels(self, labels, task_ids):
        # Every permutation uses the same digit labels.
        return labels

    def protocol(self):
        return {"name": self.name, "permutation_seed": self.permutation_seed,
                "permutations": [p.tolist() for p in self._permutations],
                "task_classes": [list(range(10)) for _ in range(self.num_tasks)],
                "task_order": list(range(self.num_tasks)),
                "transform": repr(self._train_full.transform)}


class _PermutedDataset(Dataset):
    def __init__(self, base_dataset, permutation: np.ndarray):
        self.base = base_dataset
        self.perm = torch.from_numpy(permutation).long()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # Flatten, permute, and restore the original image shape.
        flat = img.view(-1)
        flat = flat[self.perm]
        img = flat.view(img.shape)
        return img, label
