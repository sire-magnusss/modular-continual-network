# Modular Continual Network (MCN)

A PyTorch research prototype for studying shared representations and task-specific capacity in continual learning. MCN combines a shared CNN encoder, a CNN module for each task, learned feature fusion, and separate classifier heads.

**Evidence status:** the headline numbers below are archived single-run results. They have not been revalidated with the corrected evaluation, freezing, and gate ablation code. They are not multi-seed estimates or state-of-the-art claims.

## Architecture and evaluation setting

- The primary evaluation is **task-incremental**: the correct task ID selects the head at test time. Five-task Split-CIFAR-10 consists of five binary problems; its average accuracy is not unrestricted ten-class accuracy.
- The shared encoder is trained on Task 0. CIFAR freezes the entire base afterward; Permuted MNIST freezes the low encoder and adapts the upper encoder at 0.1 times the task learning rate. Frozen paths keep BatchNorm statistics fixed and dropout disabled, including when the trainer calls `train()` again.
- Each task module has a learned scalar sigmoid gate. A task-specific router uses two attention weights to blend shared and task features, followed by fusion and a residual connection. Completed task components are frozen.
- All task modules are allocated at construction using `num_tasks`. Capacity scales linearly with the configured task count; this is not online discovery of new tasks. The default CIFAR-10 model adds 1,469,381 parameters per task and has 10,591,769 total parameters for five tasks, versus 3,249,994 for the shared baseline backbone.
- MCNv2 adds cross-attention over completed modules. Its implementation is available, but a performance advantage over MCN has not been validated.
- Optional entropy-based routing evaluates task-free predictions in the **global label space** and reports head-selection accuracy separately. For permuted MNIST, the ten digit labels are shared across tasks, so a correct digit from another head is still a correct class prediction. Entropy routing is an uncalibrated heuristic.

## Archived results

All forgetting and backward-transfer differences below are in **percentage points**. These files predate the current correctness fixes; use them as historical reports, not expected outputs from the current code.

| Benchmark | MCN average accuracy | Forgetting | Evidence |
|---|---:|---:|---|
| Split-CIFAR-10, 5 tasks | 92.72% | 0.09 pp | [Final log](results/final_cifar10.txt), 10 epochs/task; BWT -0.01 pp |
| Permuted MNIST, 5 tasks | 95.91% | 1.24 pp | [Final log](results/final_mnist.txt), 10 epochs/task; BWT -1.13 pp |
| Split-CIFAR-100, 20 tasks | 75.1% | 1.5 pp | Previously reported README values and [plots](results/plots/SplitCIFAR100_summary.png); no matching raw text log committed |

The final CIFAR-10 log reports Naive 50.7%, EWC 60.8%, and PackNet-style 77.0% average accuracy. A [different archived run](results/cifar10_5task.txt) reports EWC 87.3% and MCN 92.8%. Their configurations and generating commits are not fully recorded, so these differences should not be interpreted as controlled replications. On the final Permuted MNIST run, EWC's reported 96.8% exceeds MCN's 95.91%. The previous README also reported HAT at 59.4% on CIFAR-10, without a corresponding text log in this repository.

The [legacy three-task ablation](results/ablation_3task.txt) reports full MCN 89.17%, no router 87.67%, no gate 89.73%, and base only 71.75%. The old no-gate implementation also changed MLP size and dropout, so it did not isolate the gate. The current variant removes only the scalar gates and preserves all other layers and paired initialization. It needs new results. The no-router variant still changes parameter count and is not a capacity-matched causal test.

## Install and test

Use Python 3.12 for the CPU test environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a CPU-only Linux installation, install `torch` and `torchvision` from `https://download.pytorch.org/whl/cpu` before installing `requirements.txt`. The test suite uses synthetic inputs and does not download datasets. It checks frozen state, checkpoint restoration, global-label scoring, controlled gate ablation, metric arithmetic, seed replay, and the actual runner's artifacts. Synthetic test accuracies are not research results.

Dependencies specify minimum versions for portability. Each experiment records the exact installed package versions, Python, device, and thread count. Results are not guaranteed identical across PyTorch releases or hardware backends.

## Run experiments

```bash
# Short real-data pipeline check (not a headline benchmark)
python main.py --benchmark cifar10 --methods mcn --tasks 2 --epochs 1 \
  --seeds 42 --device cpu --threads 2 --deterministic --task-free

# Proposed primary experiment: five training seeds, fixed class order
python main.py --benchmark cifar10 --methods naive ewc packnet hat mcn \
  --epochs 10 --seeds 1 2 3 4 5 --output-dir runs

# Controlled gate ablation; other fusion variants remain exploratory
python main.py --benchmark cifar10 --methods mcn mcn_no_gate \
  --epochs 10 --seeds 1 2 3 4 5

# Shared digit labels; permutation seed is separate from training seeds
python main.py --benchmark mnist --methods ewc mcn --epochs 10 \
  --seeds 1 2 3 4 5 --permutation-seed 42 --task-free
```

`--device auto` prefers MPS, then CUDA, then CPU. `--deterministic` requires deterministic PyTorch algorithms and raises for unsupported operations; seeded execution without this flag is not a determinism guarantee. Each method is reseeded before initialization, making its run independent of CLI method order. New runs use unique directories and do not overwrite historical plots or logs.

An experiment directory contains:

```text
experiment.json            # arguments, protocol, source commit/hashes, environment
source.zip                 # exact source used, including uncommitted code
summary.json               # mean and sample std across completed seeds per method
seed_42/mcn/run.json        # status, metrics, parameter count, runtime, checkpoint size
seed_42/mcn/train.log
seed_42/mcn/final_model.pt
seed_42/plots/              # unless --no-plots
```

JSON metrics use fractions; multiply by 100 for accuracy percent and forgetting/BWT percentage points. Unmeasured matrix entries are `null`. Standard deviation is `null` for a single seed. A failed run retains its error and log. `summary.json` contains completed runs only; inspect its seed list and count before reporting. Run artifacts are ignored by git; deliberately archive the selected full experiment directory alongside any future paper or release.

Final checkpoints support inference, not mid-training resumption:

```python
import torch
from main import build_model

checkpoint = torch.load("runs/EXPERIMENT/seed_42/mcn/final_model.pt",
                        map_location="cpu", weights_only=True)
model = build_model(**checkpoint["model_config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
# model(images, task_id=0) supplies the known task ID.
# model.predict_task_free(images) returns (task-local labels, selected task IDs).
# Use the matching benchmark.to_global_labels() to interpret task-free predictions.
```

The current MCN state dict includes lifecycle metadata. Legacy checkpoints lacking that metadata need explicit migration; do not silently load them as trained models.

## Remaining validation before strong research claims

1. Run the corrected implementation over multiple seeds with a held-out validation split for tuning. Current loaders expose train/test splits, not a tuning split.
2. Validate the prototype baselines. In particular, PackNet's forward pass currently does not apply its stored task masks, and its pruning/retraining protocol needs review. EWC's batch-gradient Fisher approximation needs validation and tuning.
3. Compare capacity and compute fairly. Add an independent-network-per-task control; report parameter/storage growth and inference cost as well as accuracy. MNIST currently compares CNN-based MCN against MLP-based shared baselines.
4. Vary class/task order; the first task determines the frozen base representation.
5. Re-run all ablations, calibrate task-free routing, and archive full accuracy matrices and final checkpoints. Fixing the scorer does not validate the heuristic.

See [the research evidence note](docs/research_evidence.md) for defensible application wording and the boundary between implemented methods and validated claims.

## Project structure

- `benchmarks/`: task definitions, loaders, and global label mappings.
- `models/`: MCN, MCNv2, variants, baselines, and frozen-state lifecycle.
- `trainers/`: sequential training loops.
- `utils/`: metrics, global-label evaluation, experiment records, and plots.
- `tests/`: download-free correctness and runner checks.
- `results/`: archived logs and plots.
- `paper/`: archived preprint PDF; its claims require the same validation caveats.

## Citation

```bibtex
@misc{makgasane_mcn,
  title        = {Modular Continual Network},
  author       = {Magnus Tiiso Makgasane},
  howpublished = {\url{https://github.com/sire-magnusss/modular-continual-network}},
  note         = {Research software and archived preprint; preliminary experimental results}
}
```

MIT licensed. Cite the exact commit used in an experiment.
