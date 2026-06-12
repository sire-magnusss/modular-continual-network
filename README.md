# Modular Continual Network (MCN)

A novel neural network architecture for continual learning that achieves near-zero catastrophic forgetting by growing dedicated modular capacity per task — instead of competing over a fixed set of weights.

---

## Results

### Split-CIFAR-10 — 5 Tasks

![CIFAR-10 Summary](results/plots/SplitCIFAR10_summary.png)

![CIFAR-10 Accuracy Matrix](results/plots/SplitCIFAR10_accuracy_matrix.png)

### Split-CIFAR-100 — 20 Tasks (Hard Benchmark)

![CIFAR-100 Summary](results/plots/SplitCIFAR100_summary.png)

![CIFAR-100 Accuracy Matrix](results/plots/SplitCIFAR100_accuracy_matrix.png)

### Permuted MNIST — 5 Tasks

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

> On Permuted MNIST, EWC is competitive — tasks are structurally identical (all digit images), so soft regularization is sufficient. MCN's advantage is clearest on visually diverse tasks.

---

## How It Works

```
Input ──► base_low  [frozen after Task 0]  ──► base_high ──► base_feat (512d)
    │      Blocks 1+2: edges & textures          Block 3+FC         │
    │                                                                │
    └──► TaskModule[t] ──────────────────────────► task_feat ───────┤
          Lightweight CNN adapter (new per task)      (256d)         │
                                                                     ▼
                                                               Router[t]
                                                          (per-sample attention)
                                                                     │
                                                                     ▼
                                                              Head[t] → logits
```

**The core idea** is simple: instead of all tasks fighting over the same weights (which causes forgetting), each new task gets its own dedicated module. The base encoder trains on Task 0 and then freezes — those representations never degrade. New tasks grow their own capacity and can never touch old weights.

**Why it beats the alternatives:**

- **EWC** adds a soft penalty to protect important weights — but the penalty accumulates across tasks and eventually chokes new learning. At 20 tasks on CIFAR-100 it drops to 66% accuracy.
- **PackNet** uses hard binary masks per task — zero forgetting by construction, but the network physically runs out of free parameters. It collapses to 56% at 20 tasks.
- **HAT** learns which capacity to allocate via gradient — smarter than PackNet, but still hits the same fixed-capacity wall.
- **MCN** has no capacity limit. Each new CIFAR task adds ~1.47M parameters in the current implementation. No competition, no wall.

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

# The hard benchmark — 20 tasks on CIFAR-100
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

```
modular-continual-network/
├── benchmarks/
│   ├── split_cifar10.py        5-task CIFAR-10
│   ├── split_cifar100.py       20-task CIFAR-100
│   └── permuted_mnist.py       5-task Permuted MNIST
├── models/
│   ├── backbone.py             CNN/MLP backbones for baselines
│   ├── ewc.py                  Elastic Weight Consolidation
│   ├── packnet.py              PackNet (prune & freeze)
│   ├── hat.py                  HAT (hard attention to task)
│   ├── mcn.py                  Modular Continual Network (ours)
│   ├── mcn_v2.py               MCN v2 with cross-task attention
│   └── mcn_ablations.py        MCN ablation variants
├── trainers/
│   ├── naive_trainer.py        Sequential SGD baseline
│   ├── ewc_trainer.py          EWC training loop
│   ├── packnet_trainer.py      PackNet two-phase training
│   ├── hat_trainer.py          HAT with temperature annealing
│   └── mcn_trainer.py          MCN trainer with task-free inference
├── paper/
│   └── Modular Continual Network preprint by MAGNUS MAKGASANE.pdf  Full paper
├── utils/
│   ├── device.py               MPS / CUDA / CPU detection
│   ├── metrics.py              AA, BWT, Forgetting Measure
│   └── visualization.py        Result plots
├── results/
│   └── plots/                  Generated PNG plots
└── main.py                     CLI experiment runner
```

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
