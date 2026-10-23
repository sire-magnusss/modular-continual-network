"""Task-free scoring in the benchmark's global label space."""

import torch


@torch.no_grad()
def evaluate_task_free(model, benchmark, device):
    """Score class predictions and report head-selection accuracy separately.

    CIFAR tasks have disjoint classes; identical local indices in two heads
    represent different classes. MNIST permutations share the same ten labels,
    so a correct digit remains correct even when a different head is selected.
    """
    model.eval()
    rows = []
    for task_id in range(benchmark.num_tasks):
        correct = task_correct = total = 0
        for x, y in benchmark.get_test_loader(task_id):
            x, y = x.to(device), y.to(device)
            labels, tasks = model.predict_task_free(x)
            predicted = benchmark.to_global_labels(labels, tasks)
            expected = benchmark.to_global_labels(y, torch.full_like(y, task_id))
            correct += (predicted == expected).sum().item()
            task_correct += (tasks == task_id).sum().item()
            total += y.numel()
        if not total:
            raise ValueError(f"Task {task_id} has no test examples.")
        rows.append({"task_id": task_id, "samples": total,
                     "correct": correct, "correct_task": task_correct,
                     "accuracy": correct / total,
                     "task_selection_accuracy": task_correct / total})
    total = sum(row["samples"] for row in rows)
    return {
        "label_space": "global",
        "samples": total,
        "accuracy": sum(row["correct"] for row in rows) / total,
        "task_selection_accuracy": sum(row["correct_task"] for row in rows) / total,
        "per_task": rows,
    }
