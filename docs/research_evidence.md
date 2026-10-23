# Research evidence and claim boundaries

This note separates source-code evidence from archived performance reports. The
historical audit examined commit `647626194dd0bc4cb739722b413d702a2ed80c1d`.
The correctness changes in the current branch do not reproduce or certify the old scores.

## Supported by implementation

- Python/PyTorch implementation of shared CNN features, task-specific CNN modules,
  sigmoid gates, learned feature fusion, and per-task output heads.
- Task-incremental training/evaluation on Split-CIFAR-10, Split-CIFAR-100, and
  Permuted MNIST; prototype EWC, PackNet-style, HAT, and naive baselines.
- MCNv2 cross-attention over prior task-module features and exploratory ablations.
- Accuracy-matrix evaluation with average accuracy, backward transfer, and forgetting.
- Current code adds global-label task-free scoring, frozen-state lifecycle checks,
  controlled gate removal, seed controls, run manifests, and inference checkpoints.

## Supported as historical reports, not independently reproduced findings

| Claim | Artifact | Required qualification |
|---|---|---|
| 92.72% AA; 0.09 pp forgetting; -0.01 pp BWT | `results/final_cifar10.txt` | Single recorded five-task Split-CIFAR-10 run, 10 epochs/task, task ID known |
| 95.91% AA; 1.24 pp forgetting | `results/final_mnist.txt` | Single five-task Permuted MNIST run; EWC performs better in this log |
| 75.1% AA; 1.5 pp forgetting on CIFAR-100 | Historical README and plots | No corresponding text log in the audited tree; weaker provenance |
| 15.74 pp lead over PackNet-style in final CIFAR log | 92.72% minus 76.98% | Comparison with this prototype, not established superiority over the published method |

Other archived runs report different scores, including much stronger EWC performance.
Their missing configuration/seed provenance prevents treating them as controlled replicates.
The archived preprint is not evidence of peer-reviewed publication or acceptance.

## Copy-ready wording

### CV project entry

Developed Modular Continual Network, a PyTorch framework combining shared CNN
features with task-specific modules and learned feature routing; recorded 92.72%
average accuracy and 0.09 percentage-point forgetting in a single five-task,
task-incremental Split-CIFAR-10 experiment.

Use the historical result clause only with its qualifications. Until reruns are
complete, a methods-only entry is also appropriate:

Implemented and open-sourced a modular continual-learning framework in PyTorch,
with task-specific CNN modules, learned feature fusion, baseline comparisons,
and component ablations across three benchmark settings.

### Research profile

My continual-learning research explores how task-specific capacity and shared
representations affect the stability-plasticity trade-off. I implemented modular
architectures, baseline comparisons, and ablation experiments, with validation
focused on reproducibility, capacity fairness, and task-free evaluation.

## Claims to defer

- State-of-the-art performance, independent reproducibility, or a general solution
  to catastrophic forgetting.
- 92.72% unrestricted ten-class CIFAR-10 accuracy.
- Validated task-free performance from the historical local-label scorer.
- A causal benefit from the gate or capacity-unmatched router ablation.
- A demonstrated advantage for MCNv2, parameter efficiency, or compute efficiency.
- Automatically discovered tasks or online allocation: modules are preallocated.

## Next experiment to answer a useful scientific question

Does learned fusion of shared and task-specific representations improve accuracy
per unit of capacity over simpler task-isolated models?

Use the corrected code, an explicit validation split, multiple seeds and task orders,
validated baselines, an independent-network-per-task control, and a table reporting
accuracy, forgetting, parameter/storage growth, training time and inference cost.
Keep legacy scores separate from new measurements. Define the novelty relative to
prior modular/expansion methods only after a dedicated literature review.
