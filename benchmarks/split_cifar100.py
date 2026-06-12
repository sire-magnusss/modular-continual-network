"""Split-CIFAR-100 benchmark."""

from typing import List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

TASK_CLASSES = [list(range(i * 5, (i + 1) * 5)) for i in range(20)]


class SplitCIFAR100:
    """Build train/test loaders for twenty five-class CIFAR-100 tasks."""

    num_tasks = 20
    num_classes_per_task = 5
    input_shape = (3, 32, 32)
    name = "SplitCIFAR100"

    def __init__(self, data_dir: str = "./data", batch_size: int = 128,
                 num_workers: int = 0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        self._train_full = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        self._test_full = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    def _make_remapped_subset(self, dataset, original_classes: List[int]):
        remap = {orig: new for new, orig in enumerate(original_classes)}
        indices = [i for i, (_, label) in enumerate(dataset)
                   if label in original_classes]

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
        subset = self._make_remapped_subset(self._train_full, TASK_CLASSES[task_id])
        return DataLoader(subset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def get_test_loader(self, task_id: int) -> DataLoader:
        subset = self._make_remapped_subset(self._test_full, TASK_CLASSES[task_id])
        return DataLoader(subset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def task_description(self, task_id: int) -> str:
        classes = TASK_CLASSES[task_id]
        return f"Task {task_id:02d}: CIFAR-100 classes {classes[0]}-{classes[-1]}"
