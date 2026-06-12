"""Split-CIFAR-10 benchmark."""

from typing import List, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

TASK_CLASSES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class SplitCIFAR10:
    """Build train/test loaders for five two-class CIFAR-10 tasks."""

    num_tasks = 5
    num_classes_per_task = 2
    input_shape = (3, 32, 32)
    name = "SplitCIFAR10"

    def __init__(self, data_dir: str = "./data", batch_size: int = 128,
                 num_workers: int = 0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        self._train_full = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        self._test_full = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    def _filter_indices(self, dataset, original_classes: List[int]) -> Tuple[list, dict]:
        """Return indices where label is in original_classes, plus a remapping dict."""
        remap = {orig: new for new, orig in enumerate(original_classes)}
        indices = [i for i, (_, label) in enumerate(dataset)
                   if label in original_classes]
        return indices, remap

    def _make_remapped_subset(self, dataset, original_classes: List[int]):
        """Returns a Dataset with remapped labels (0, 1 instead of original class IDs)."""
        indices, remap = self._filter_indices(dataset, original_classes)

        class RemappedSubset:
            def __init__(self, base, idxs, mapping):
                self.base = base
                self.idxs = idxs
                self.mapping = mapping

            def __len__(self):
                return len(self.idxs)

            def __getitem__(self, idx):
                img, label = self.base[self.idxs[idx]]
                return img, self.mapping[label]

        return RemappedSubset(dataset, indices, remap)

    def get_train_loader(self, task_id: int) -> DataLoader:
        classes = TASK_CLASSES[task_id]
        subset = self._make_remapped_subset(self._train_full, classes)
        return DataLoader(subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def get_test_loader(self, task_id: int) -> DataLoader:
        classes = TASK_CLASSES[task_id]
        subset = self._make_remapped_subset(self._test_full, classes)
        return DataLoader(subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def task_description(self, task_id: int) -> str:
        classes = TASK_CLASSES[task_id]
        names = " vs ".join(CLASS_NAMES[c] for c in classes)
        return f"Task {task_id}: {names}"
