# Modular Continual Network (MCN)

PyTorch experiments for a modular continual-learning model that grows task-specific capacity instead of training every task through the same shared weights.

These results are from task-incremental experiments where task identity is known at inference time.

---

## Results

Reported results are deterministic single-run measurements; multi-seed mean/std evaluation and stronger baseline tuning are future work.

### Split-CIFAR-10 - 5 Tasks

![CIFAR-10 Summary](results/plots/SplitCIFAR10_summary.png)

![CIFAR-10 Accuracy Matrix](results/plots/SplitCIFAR10_accuracy_matrix.png)

### Split-CIFAR-100 - 20 Tasks

![CIFAR-100 Summary](results/plots/SplitCIFAR100_summary.png)

![CIFAR-100 Accuracy Matrix](results/plots/SplitCIFAR100_accuracy_matrix.png)

### Permuted MNIST - 5 Tasks

![MNIST Summary](results/plots/PermutedMNIST_summary.png)

![MNIST Accuracy Matrix](results/plots/PermutedMNIST_accuracy_matrix.png)

---

## Numbers

### Split-CIFAR-10 (5 tasks)

| Method | Avg Accuracy | Forgetting |
|--------|:---:|:---:|
| Naive | 50.7% | 55.6% |
| EWC | 60.8% | 39.2% |
| HAT | 59.4% | 43.7% |
| PackNet | 77.0% | 9.3% |
| **MCN (ours)** | **92.7%** | **0.1%** |

### Split-CIFAR-100 (20 tasks)

| Method | Avg Accuracy | Forgetting |
|--------|:---:|:---:|
| Naive | 23.8% | 64.4% |
| PackNet | 56.2% | 4.5% |
| EWC | 66.0% | 11.3% |
| **MCN (ours)** | **75.1%** | **1.5%** |

### Permuted MNIST (5 tasks)

| Method | Avg Accuracy | Forgetting |
|--------|:---:|:---:|
| Naive | 87.5% | 12.8% |
| PackNet | 95.4% | 3.0% |
| MCN (ours) | 95.9% | 1.2% |
| EWC | **96.8%** | **1.0%** |

> On Permuted MNIST, EWC is competitive because the tasks share the same digit structure. MCN's strongest results are on the CIFAR task splits.

---

## How It Works

MCN uses four pieces:

- A base encoder trained on Task 0 and then frozen.
- A task-specific CNN adapter for each task.
- A task-specific router that blends base and adapter features.
- A task-specific output head.

This avoids a fixed shared-capacity ceiling by growing capacity per task, at the cost of linear model growth. Each new task adds approximately 1.47M parameters in the CIFAR configuration.

---

## Ablation Study

Tested on 3-task Split-CIFAR-10 to isolate each component's contribution:

| Variant | Avg Accuracy | Forgetting | What it shows |
|---------|:---:|:---:|---|
| MCN (full) | 89.2% | 0.3% | Full architecture |
| No Router | 87.7% | 0.1% | Attention router adds +1.5% accuracy |
| No Gate | 89.7% | 0.0% | Gate stabilizes early training |
| Base Only | 71.7% | 0.4% | Task modules are essential (+17.5%) |

The task modules are the critical component. Without them (Base Only), the frozen base encoder alone isn't enough. The router's per-sample attention blending adds a further boost on top.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# All methods on Split-CIFAR-10
python main.py --benchmark cifar10 --methods naive ewc packnet hat mcn --epochs 10

# 20 tasks on CIFAR-100
python main.py --benchmark cifar100 --methods naive ewc packnet mcn --epochs 10

# Permuted MNIST
python main.py --benchmark mnist --methods naive ewc packnet mcn --epochs 10

# MCN v2 (cross-task knowledge sharing)
python main.py --benchmark cifar10 --methods mcn mcn_v2 --epochs 10

# Ablation study
python main.py --methods mcn mcn_no_router mcn_no_gate mcn_base_only --tasks 3 --epochs 5
```

---

## Project Structure

- `benchmarks/`: Split-CIFAR-10, Split-CIFAR-100, and Permuted MNIST loaders.
- `models/`: MCN, MCN v2, ablations, and baseline models.
- `trainers/`: training loops for each method.
- `utils/`: device selection, metrics, and plotting.
- `results/`: saved logs and generated plots.
- `paper/`: compiled preprint PDF.
- `main.py`: command-line experiment runner.

---

## Metrics

**Average Accuracy (AA)** — mean test accuracy across all tasks after the final task. Higher is better.

**Backward Transfer (BWT)** — how much training new tasks hurt old ones. More negative = more forgetting.

**Forgetting Measure (FM)** — average drop from peak accuracy per task. Lower is better.

---

## Current Limitations

- Headline results use task-incremental evaluation, where the task ID is known at test time.
- Results are single-run experiments; multi-seed mean/std reporting is future work.
- Model size grows linearly as new task modules are added.
- Baseline implementations are research prototypes intended for comparison and study.

---

## Hardware

Developed and tested on Apple M4 MacBook using MPS (Metal Performance Shaders) acceleration. Automatically falls back to CUDA or CPU.

---

## Citation

```bibtex
@misc{mcn2025,
  title   = {Modular Continual Network: Growing Capacity for Near-Zero Catastrophic Forgetting},
  author  = {Magnus Tiiso Makgasane},
  year    = {2025},
  note    = {Preprint}
}
```
