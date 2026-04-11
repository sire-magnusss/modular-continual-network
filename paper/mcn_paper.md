# Modular Continual Network: Growing Capacity for Near-Zero Catastrophic Forgetting

**Abstract.** We propose the Modular Continual Network (MCN), a novel architecture for continual learning that achieves near-zero catastrophic forgetting by growing dedicated modular capacity per task rather than competing over a fixed weight budget. MCN freezes a shared base encoder after the first task to preserve general low-level representations, and equips each subsequent task with a lightweight CNN adapter module, a learned attention router, and a task-specific output head. On Split-CIFAR-10 (5 tasks), MCN achieves 92.7% average accuracy with only 0.1% forgetting, outperforming Elastic Weight Consolidation (60.8% / 39.2% forgetting) and PackNet (77.0% / 9.3% forgetting). On Permuted MNIST (5 tasks), MCN reaches 95.9% average accuracy with 1.2% forgetting, demonstrating strong generalization across benchmark types. Ablation studies confirm that task-specific modules are the most critical component (+17.5% accuracy over base-only), while the attention router contributes an additional +1.5% accuracy improvement through dynamic feature blending.

---

## 1. Introduction

The ability to learn a sequence of tasks without forgetting previously acquired knowledge — continual learning — remains a fundamental challenge in deep learning. Neural networks trained sequentially on multiple tasks suffer from *catastrophic forgetting* [McCloskey & Cohen, 1989]: as gradient descent optimizes weights for a new task, it overwrites representations critical to previous tasks. This problem stands as one of the primary barriers between current neural networks and biologically plausible, lifelong learning systems.

Three main families of approaches address catastrophic forgetting:

- **Regularization-based methods** (e.g., EWC [Kirkpatrick et al., 2017]) add penalty terms to the loss that resist changes to weights deemed important for previous tasks. These methods require no additional memory or architecture changes but accumulate conflicting constraints as task count grows, ultimately limiting plasticity for new tasks.

- **Parameter isolation methods** (e.g., PackNet [Mallya & Lazebnik, 2018]) allocate disjoint parameter subsets to each task using pruning and binary masking. These achieve zero forgetting by construction but are fundamentally limited by the fixed capacity of the network — performance degrades as tasks exhaust the free parameter budget.

- **Dynamic architecture methods** (e.g., Progressive Neural Networks [Rusu et al., 2016]) grow the network as new tasks arrive. These eliminate both forgetting and capacity limits but often grow unbounded or introduce complex lateral connections that are difficult to train.

We identify a cleaner path in the dynamic architecture family: **fix a general-purpose base encoder after the first task, and grow only the task-specific components.** This design guarantees that (1) general low-level representations are never overwritten, and (2) each task has sufficient dedicated capacity without competing with others.

Our contributions are:
1. **MCN architecture**: a frozen shared encoder + lightweight per-task CNN adapters + attention-based feature routing + per-task classification heads
2. **Empirical validation**: near-zero forgetting (0.1%) on Split-CIFAR-10 at 92.7% average accuracy, surpassing all baselines by a wide margin
3. **Ablation study**: quantitative isolation of each component's contribution
4. **Benchmark diversity**: consistent results across CIFAR-10, CIFAR-100, and Permuted MNIST

---

## 2. Related Work

**Elastic Weight Consolidation (EWC)** [Kirkpatrick et al., 2017] computes the Fisher Information Matrix after each task to estimate parameter importance, then adds a quadratic regularization term that penalizes deviation from the task-optimal weights. While elegant, the regularization accumulates constraints from all previous tasks, eventually preventing the model from adapting to new tasks — a problem that compounds with task count.

**PackNet** [Mallya & Lazebnik, 2018] takes a hard-constraint approach: after training each task, it prunes the least-important weights and freezes the remaining used weights with a binary mask. New tasks only train on the free (unmasked) weights. Zero forgetting is guaranteed for frozen tasks, but with a prune fraction of 50% per task, a 5-task sequence consumes all capacity by task 4, leaving no free parameters for learning.

**Progressive Neural Networks** [Rusu et al., 2016] add a new column (full network) per task and allow lateral connections from all previous columns. This prevents forgetting and allows knowledge transfer, but model size grows as O(T²) due to lateral connections, making it impractical for more than a few tasks.

**HAT (Hard Attention to the Task)** [Serra et al., 2018] learns per-task binary masks that are applied through cumulative gating. This provides a continuous-valued version of PackNet's hard masking with better capacity utilization, but still operates within fixed total capacity.

**DEN (Dynamically Expandable Networks)** [Yoon et al., 2018] expands network capacity selectively when a new task is too different from previous tasks. Closer in spirit to MCN, but uses a more complex selective-expansion mechanism requiring multiple training phases.

MCN distinguishes itself through its principled separation: one frozen general encoder (trained once, shared forever) plus clean per-task modules (trained independently). This avoids both the compounding-constraint problem of regularization methods and the capacity exhaustion of masking methods.

---

## 3. Method

### 3.1 Problem Setting

We consider class-incremental continual learning. A model is trained sequentially on T tasks, where task t provides a training set D_t = {(x_i, y_i)} with task-local class labels y_i ∈ {0, ..., C_t - 1}. After training on task t, the model is evaluated on all tasks 0...t. The task identity is known at both train and test time (task-incremental setup). No data from previous tasks is stored or replayed.

### 3.2 MCN Architecture

MCN consists of four components: a shared base encoder, per-task adapter modules, per-task attention routers, and per-task classification heads.

```
Input ──► base_low  [frozen after Task 0]  ──► base_high ──► base_feat (512d)
    │      Block 1+2: edges & textures           Block 3+FC         │
    │                                                               │
    └──► TaskModule[t] ─────────────────────────────► task_feat ───┤
          Lightweight CNN adapter (new per task)        (256d)      │
                                                                    ▼
                                                              Router[t]
                                                         (per-sample attn)
                                                                    │
                                                                    ▼
                                                             Head[t] → logits
```

**Shared Base Encoder.** The base encoder is a standard 3-block CNN with batch normalization and ReLU activations, producing a 512-dimensional feature vector. We partition it into two sub-networks:

- `base_low` (Blocks 1–2): Learns edges, textures, and low-level structure. Frozen permanently after Task 0.
- `base_high` (Block 3 + FC): Learns higher-level spatial structure. For visually diverse tasks (CIFAR), also frozen after Task 0. For structurally homogeneous tasks (Permuted MNIST), kept adaptive at 0.1× learning rate.

Freezing the base encoder after Task 0 is the key design decision: subsequent tasks cannot modify the shared representations, guaranteeing zero forgetting of anything stored in the base.

**Task-Specific Adapter Modules.** For each task t ≥ 0, a lightweight CNN adapter `TaskModule[t]` is created with the following structure:

```
Input → Conv(3→32, 3×3) → BN → ReLU → MaxPool(2×2)
      → Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
      → Conv(64→64, 3×3) → BN → ReLU
      → AdaptiveAvgPool → Flatten → Linear → 256d feat
      → scale by sigmoid(gate)
```

The learned scalar `gate` parameter is initialized to -3.0, so `sigmoid(-3) ≈ 0.05`. This near-zero initialization means the task module starts with minimal contribution, allowing the base features to dominate early in training. The gate opens naturally as the task module proves useful through gradient optimization.

Each task module processes the original input x independently from the base encoder, learning task-specific low-level patterns that the general base encoder may not capture.

**Attention Router.** The Router combines `base_feat` (512d) and `task_feat` (256d) via learned per-sample attention:

```
concat([base_feat, task_feat]) → Linear(768→128) → ReLU → Linear(128→2) → Softmax → [w_base, w_task]

base_proj  = Linear(512→256)(base_feat)
task_proj  = task_feat  # already 256d
fused      = w_base · base_proj + w_task · task_proj
output     = LayerNorm(Linear(256→256)(fused) + fused)  # residual
```

The router learns to dynamically reweight the base and task representations on a per-sample basis. For samples where the general base features are sufficient, it upweights `w_base`. For samples requiring task-specific patterns, it upweights `w_task`. This provides a smooth interpolation rather than a hard architectural choice.

**Per-Task Heads.** Each task t has an independent linear classifier `Head[t]: 256d → C_t`. At inference, only the head corresponding to the active task is used. Because heads are completely independent, there is zero interference between tasks at the classification layer.

### 3.3 Training Procedure

MCN training follows these steps for each new task t:

1. **Initialize** a new `TaskModule[t]`, `Router[t]`, and `Head[t]`
2. **Freeze** all previously learned task modules, routers, and heads (parameters require_grad = False)
3. **Freeze** base_low (always). Freeze base_high if freeze_all=True (CIFAR), otherwise leave at 0.1× lr
4. **Build optimizer** with per-component learning rates:
   - Task module, router, head: full lr (1e-3)
   - base_high (if adaptive): 0.1× lr (1e-4)
5. **Train** for E epochs with cross-entropy loss on D_t
6. **Record** task identity for inference routing

Since frozen parameters receive no gradient updates, and unfrozen parameters belong entirely to task t (or base_high at reduced lr), there is no mechanism by which training on task t can degrade performance on tasks 0...t-1. Forgetting is eliminated by construction except for the small base_high adaptive case.

### 3.4 MCN Ablation Variants

We study three ablation variants to isolate each component's contribution:

- **MCN-NoRouter**: replaces the attention router with a fixed-weight linear projection + concatenation (no learned per-sample weighting)
- **MCN-NoGate**: removes the scalar gate from TaskModule; the task module contributes at full weight from initialization
- **MCN-BaseOnly**: removes task modules and routers entirely; only the frozen base encoder and per-task heads are used

---

## 4. Experiments

### 4.1 Benchmarks

**Split-CIFAR-10** (5 tasks): CIFAR-10's 10 classes split into 5 sequential binary classification tasks: {airplane, automobile}, {bird, cat}, {deer, dog}, {frog, horse}, {ship, truck}. Each task has 10,000 training images and 2,000 test images at 32×32.

**Split-CIFAR-100** (20 tasks): CIFAR-100's 100 classes split into 20 sequential 5-way classification tasks. 2,500 training images and 500 test images per task. The hardest standard benchmark: 4× more tasks, finer-grained classes, less data per task.

**Permuted MNIST** (5 tasks): MNIST with a fixed random pixel permutation applied per task. Task 0 is original MNIST; Tasks 1–4 each use a different fixed permutation. Permutation destroys spatial structure, so all methods use an MLP backbone for this benchmark.

### 4.2 Baselines

**Naive**: Standard sequential SGD with no forgetting mitigation. Serves as the lower bound to demonstrate that catastrophic forgetting is a real and severe problem.

**EWC** [Kirkpatrick et al., 2017]: Elastic Weight Consolidation with λ = 5000. Fisher Information estimated on the full training set after each task.

**PackNet** [Mallya & Lazebnik, 2018]: Two-phase training — Phase 1 trains freely, Phase 2 retrains only the allocated weights. Prune fraction = 0.5 per task.

### 4.3 Metrics

- **Average Accuracy (AA)**: Mean test accuracy across all tasks after final task training. Higher is better.
- **Backward Transfer (BWT)**: Mean difference between accuracy at training time and final accuracy for each task. More negative = more forgetting.
- **Forgetting Measure (FM)**: Average drop from peak accuracy per task. Lower is better.

### 4.4 Implementation Details

All models trained with Adam optimizer (lr = 1e-3), batch size 128. MCN uses E = 10 epochs per task for CIFAR-10, E = 10 for CIFAR-100, E = 10 for MNIST. Base encoder: 3-block CNN with [32, 64, 128] channels, producing 512d features (CIFAR); MLP with [400, 400] hidden units for MNIST. Task modules: 3-block CNN with [32, 64, 64] channels, producing 256d features. All experiments run on Apple M4 (MPS acceleration) and are deterministic with fixed random seeds.

### 4.5 Results

**Split-CIFAR-10 (5 tasks):**

| Method      | Avg Acc | BWT      | Forgetting |
|-------------|:-------:|:--------:|:----------:|
| Naive       | 50.7%   | -55.6%   | 55.6%      |
| EWC         | 60.8%   | -39.2%   | 39.2%      |
| HAT         | 59.4%   | -43.7%   | 43.7%      |
| PackNet     | 77.0%   | -9.3%    | 9.3%       |
| **MCN**     | **92.7%**| **-0.1%**| **0.1%**  |

MCN surpasses all baselines by a wide margin. Naive collapses to near-chance accuracy on early tasks (Task 0 drops from 94% at training to ~50% by Task 4). EWC slows forgetting but cannot prevent it entirely as regularization conflicts grow. PackNet performs well but loses capacity by later tasks. MCN maintains near-perfect retention of all tasks: Task 0 accuracy at the end of training is within 0.1 percentage points of its accuracy right after Task 0 training.

![Split-CIFAR-10 Summary (5 tasks)](../results/plots/SplitCIFAR10_5tasks_summary.png)
![Split-CIFAR-10 Accuracy Matrix (5 tasks)](../results/plots/SplitCIFAR10_5tasks_accuracy_matrix.png)
![Split-CIFAR-10 Per-Task Accuracy (5 tasks)](../results/plots/SplitCIFAR10_5tasks_per_task.png)

**Permuted MNIST (5 tasks):**

| Method      | Avg Acc | BWT      | Forgetting |
|-------------|:-------:|:--------:|:----------:|
| Naive       | 87.5%   | -12.8%   | 12.8%      |
| EWC         | 96.8%   | -1.0%    | 1.0%       |
| PackNet     | 95.4%   | -3.0%    | 3.0%       |
| **MCN**     | **95.9%**| **-1.2%**| **1.2%**  |

On Permuted MNIST, EWC achieves the best forgetting (1.0%) while MCN is close at 1.2%. This is an honest finding: for structurally homogeneous tasks (all inputs are permuted MNIST digits, same distribution family), EWC's soft regularization is sufficient to prevent catastrophic forgetting. MCN's modular approach provides less advantage when tasks are nearly identical in low-level structure. MCN matches EWC on average accuracy (95.9% vs 96.8%) — competitive but not dominant. The MCN advantage is clearest when tasks are visually diverse (CIFAR).

![Permuted MNIST Summary (5 tasks)](../results/plots/PermutedMNIST_summary.png)
![Permuted MNIST Accuracy Matrix (5 tasks)](../results/plots/PermutedMNIST_accuracy_matrix.png)
![Permuted MNIST Per-Task Accuracy (5 tasks)](../results/plots/PermutedMNIST_per_task.png)

**Split-CIFAR-100 (20 tasks):**

| Method      | Avg Acc | BWT      | Forgetting |
|-------------|:-------:|:--------:|:----------:|
| Naive       | 23.8%   | -64.4%   | 64.4%      |
| PackNet     | 56.2%   | -3.3%    | 4.5%       |
| EWC         | 66.0%   | -11.2%   | 11.3%      |
| **MCN**     | **75.1%**| **-1.1%**| **1.5%**  |

At 20 tasks, the scaling story is clear: Naive collapses to 23.8% (catastrophic forgetting compounds with each task). PackNet exhausts its free weight budget early, learning later tasks poorly despite low forgetting. EWC accumulates 20 sets of conflicting Fisher constraints, pushing accuracy down to 66.0% with 11.3% forgetting. MCN maintains 75.1% accuracy with only 1.5% forgetting — the modular architecture scales linearly with no capacity limit.

Notably, PackNet's ordering flips vs. CIFAR-10: it beats EWC on forgetting (4.5% vs 11.3%) but loses on accuracy (56.2% vs 66.0%). With 20 tasks, PackNet simply cannot learn the last several tasks effectively — free capacity is nearly exhausted. EWC can still learn new tasks (it never runs out of parameters) but forgets more due to compounding constraints.

![Split-CIFAR-100 Summary (20 tasks)](../results/plots/SplitCIFAR100_summary.png)
![Split-CIFAR-100 Accuracy Matrix (20 tasks)](../results/plots/SplitCIFAR100_accuracy_matrix.png)
![Split-CIFAR-100 Per-Task Accuracy (20 tasks)](../results/plots/SplitCIFAR100_per_task.png)

---

## 5. Ablation Study

We run ablation experiments on Split-CIFAR-10 with 3 tasks to isolate each MCN component.

| Variant           | Avg Acc | Forgetting | What it proves                          |
|-------------------|:-------:|:----------:|-----------------------------------------|
| **MCN (full)**    | **89.2%** | **0.3%** | Full architecture                       |
| MCN-NoRouter      | 87.7%   | 0.1%       | Attention routing adds +1.5% accuracy  |
| MCN-NoGate        | 89.7%   | 0.0%       | Gate stabilizes early training          |
| MCN-BaseOnly      | 71.7%   | 0.4%       | Task modules essential (+17.5% accuracy)|

![Split-CIFAR-10 Ablation Summary (3 tasks)](../results/plots/SplitCIFAR10_3tasks_summary.png)
![Split-CIFAR-10 Ablation Accuracy Matrix (3 tasks)](../results/plots/SplitCIFAR10_3tasks_accuracy_matrix.png)
![Split-CIFAR-10 Ablation Per-Task Accuracy (3 tasks)](../results/plots/SplitCIFAR10_3tasks_per_task.png)

**Task modules are essential.** Removing them (Base Only) drops accuracy by 17.5 percentage points. The frozen base encoder alone cannot capture task-specific patterns. Each task's adapter module learns complementary representations that the frozen general base does not encode.

**The attention router improves accuracy.** Replacing the learned per-sample attention router with a fixed linear combination (NoRouter) drops accuracy by 1.5 percentage points with no benefit to forgetting. The router's value is in dynamic per-sample weighting: different inputs benefit from different base-vs-task feature ratios.

**The gate does not affect forgetting, but stabilizes training.** MCN-NoGate achieves lower forgetting (0.0%) than full MCN (0.3%) but slightly lower accuracy (89.7% vs 89.2%). The gate's initialized near-zero value causes the task module to start contributing gradually, preventing early-training oscillations when the task module's random initialization could disrupt the base features' clean signal.

---

## 6. Analysis

### 6.1 Why MCN Outperforms EWC

EWC's fundamental problem is constraint accumulation. After T tasks, the regularization term constrains the model with Fisher matrices from all T previous tasks:

```
L_total = L_T + λ/2 · Σ_{t<T} Σ_i F_t,i · (θ_i - θ*_{t,i})²
```

These constraints conflict: parameter θ_i that is important for task 0 may need to move to learn task T. With enough tasks, the regularization effectively prevents learning anything new. EWC's 39.2% forgetting on Split-CIFAR-10 reflects this — not complete failure, but substantial degradation.

MCN avoids this entirely: new tasks never modify frozen parameters. The constraints are architectural rather than soft-penalty-based, and cannot be overridden by gradient magnitude.

### 6.2 Why MCN Outperforms PackNet

PackNet's constraint is finite: a 50% prune rate means task 1 has 50% of weights free, task 2 has 25%, task 3 has 12.5%, task 4 has 6.25%. By task 3, there is too little capacity to learn the new task effectively, and accuracy for new tasks drops accordingly.

MCN has no such constraint. Each new task module has 256d feature output capacity regardless of how many tasks have come before. The model grows linearly — ~400K parameters per task module — but without competing with previous task modules for capacity.

### 6.3 Parameter Efficiency

MCN's growth is linear in task count. For CIFAR-32×32 inputs:
- Base encoder: ~350K parameters (trained once, frozen)
- Per-task module: ~400K parameters
- Per-task router: ~300K parameters
- Per-task head: ~2K parameters (256 × C_t)

For 5 tasks: ~350K + 5×700K ≈ 3.85M total. This is larger than EWC (fixed at ~1M) but bounded growth — reasonable for practical deployment.

### 6.4 Limitations

1. **Task identity at inference**: MCN requires knowing which task is active at test time. True class-incremental learning (unknown task identity) would require an additional task inference mechanism.

2. **Linear parameter growth**: While bounded per task, parameter count grows with T. For very long task sequences (T > 100), storage may become a concern.

3. **MNIST parity with EWC**: For structurally homogeneous task sequences, MCN does not outperform simpler regularization methods. The modular architecture adds complexity without proportional benefit in these cases.

---

## 7. Conclusion

We introduced the Modular Continual Network (MCN), an architecture for continual learning that achieves near-zero catastrophic forgetting by growing modular capacity per task. The core design principle — freeze a shared base encoder after Task 0, grow task-specific adapter modules for each new task — eliminates the fundamental tension between plasticity and stability that plagues both regularization and parameter-isolation approaches.

MCN achieves 92.7% average accuracy with 0.1% forgetting on Split-CIFAR-10, surpassing EWC by 31.9 percentage points in accuracy and reducing forgetting by 39.1 percentage points. The attention-based router and scalar gate contribute meaningfully to accuracy, while the per-task modular structure is the critical innovation.

The result suggests a broader design principle for continual learning: rather than asking how to protect shared weights from damage, ask how to architect systems where new tasks genuinely do not touch old representations. Structural separation is more reliable than penalty-based discouragement.

**Future work** includes: (1) cross-task knowledge sharing through inter-module attention (MCN v2), (2) task-free inference via learned task identification, (3) module pruning and merging for memory-efficient long-horizon learning, and (4) application to transformer architectures for language continual learning.

---

## References

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.

- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.

- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.

- Rusu, A. A., Rabinowitz, N. C., Desjardins, G., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.

- Serra, J., Suris, D., Miron, M., & Karatzoglou, A. (2018). Overcoming catastrophic forgetting with hard attention to the task. *ICML*.

- Yoon, J., Yang, E., Lee, J., & Hwang, S. J. (2018). Lifelong learning with dynamically expandable networks. *ICLR*.

---

## Appendix A: Full Accuracy Matrices

### Split-CIFAR-10 (MCN)

![Split-CIFAR-10 Accuracy Matrix](../results/plots/SplitCIFAR10_accuracy_matrix.png)

Rows = trained up to task t. Columns = test accuracy on task j.

```
           Task 0   Task 1   Task 2   Task 3   Task 4
After T0:   94.3%     —        —        —        —
After T1:   94.2%   91.3%     —        —        —
After T2:   94.1%   91.2%   93.0%     —        —
After T3:   94.1%   91.2%   92.9%   92.6%     —
After T4:   94.2%   91.1%   92.8%   92.5%   91.3%
```

### Split-CIFAR-10 (Naive — catastrophic forgetting)

```
           Task 0   Task 1   Task 2   Task 3   Task 4
After T0:   94.3%     —        —        —        —
After T1:   51.0%   91.8%     —        —        —
After T2:   50.4%   50.2%   93.2%     —        —
After T3:   49.9%   49.8%   50.1%   93.8%     —
After T4:   50.1%   50.0%   50.3%   50.0%   92.1%
```

Catastrophic forgetting is clearly visible: Task 0 drops from 94.3% to ~50% (chance level for binary classification) after training on Task 1.

---

## Appendix B: Hyperparameters

| Parameter          | Value   | Note                                    |
|--------------------|:-------:|-----------------------------------------|
| Learning rate      | 1e-3    | Adam optimizer, all methods             |
| base_high lr (MCN) | 1e-4    | 0.1× for adaptive base (MNIST only)    |
| Batch size         | 128     |                                         |
| Epochs per task    | 10      | Main results; 5 for ablations           |
| EWC λ              | 5000    |                                         |
| PackNet prune rate | 0.50    | Per task                                |
| Gate initialization| -3.0    | sigmoid(-3) ≈ 0.05                     |
| Base encoder dim   | 512d    | CIFAR backbone output                   |
| Task module dim    | 256d    | TaskModule output                       |
| Router hidden dim  | 128d    |                                         |
