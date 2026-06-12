"""
Visualization utilities for continual learning experiments.
Saves plots to results/plots/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_results(results: dict, benchmark_name: str):
    """
    results = {
        "Naive": MetricTracker,
        "EWC":   MetricTracker,
        "PackNet": MetricTracker,
    }
    Generates three plots and saves them.
    """
    _ensure_dir()
    _plot_accuracy_matrix(results, benchmark_name)
    _plot_per_task_accuracy(results, benchmark_name)
    _plot_summary_bar(results, benchmark_name)
    print(f"\n[viz] Plots saved to {os.path.abspath(RESULTS_DIR)}/")


def _plot_accuracy_matrix(results, benchmark_name):
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, (method, tracker) in zip(axes, results.items()):
        R = np.array(tracker.R)
        num_tasks = R.shape[0]
        mask = np.isnan(R)
        R_display = np.where(mask, 0.0, R * 100)

        im = ax.imshow(R_display, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_xlabel("Evaluated on Task")
        ax.set_ylabel("After Training Task")
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))
        ax.set_xticklabels([f"T{i}" for i in range(num_tasks)])
        ax.set_yticklabels([f"T{i}" for i in range(num_tasks)])

        for i in range(num_tasks):
            for j in range(num_tasks):
                if not mask[i][j]:
                    ax.text(j, i, f"{R_display[i][j]:.0f}", ha="center",
                            va="center", fontsize=9,
                            color="black" if R_display[i][j] > 40 else "white")

        fig.colorbar(im, ax=ax, label="Accuracy (%)")

    fig.suptitle(f"Accuracy Matrix - {benchmark_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{benchmark_name}_accuracy_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_task_accuracy(results, benchmark_name):
    # For each method, show how accuracy on each task evolves as more tasks are trained
    method_names = list(results.keys())
    num_tasks = list(results.values())[0].num_tasks

    fig, axes = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 4), sharey=True)
    if num_tasks == 1:
        axes = [axes]

    for task_id, ax in enumerate(axes):
        for m_idx, (method, tracker) in enumerate(results.items()):
            R = np.array(tracker.R)
            # x = which training step we're at, y = accuracy on task_id at that step
            xs, ys = [], []
            for train_step in range(task_id, num_tasks):
                val = R[train_step][task_id]
                if not np.isnan(val):
                    xs.append(train_step)
                    ys.append(val * 100)
            ax.plot(xs, ys, marker="o", label=method,
                    color=COLORS[m_idx % len(COLORS)], linewidth=2)

        ax.set_title(f"Task {task_id}", fontsize=11)
        ax.set_xlabel("Tasks trained")
        if task_id == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(range(task_id, num_tasks))
        ax.set_xticklabels([f"T{i}" for i in range(task_id, num_tasks)])
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-Task Accuracy Over Training - {benchmark_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{benchmark_name}_per_task.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_summary_bar(results, benchmark_name):
    method_names = list(results.keys())
    aa_vals   = [t.average_accuracy() * 100 for t in results.values()]
    bwt_vals  = [t.backward_transfer() * 100 for t in results.values()]
    fm_vals   = [t.forgetting_measure() * 100 for t in results.values()]

    x = np.arange(len(method_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, aa_vals,  width, label="Avg Accuracy (%)",      color="#4CAF50")
    bars2 = ax.bar(x,         bwt_vals, width, label="Backward Transfer (%)", color="#2196F3")
    bars3 = ax.bar(x + width, fm_vals,  width, label="Forgetting (%)",        color="#F44336")

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=12)
    ax.set_ylabel("Value (%)")
    ax.set_title(f"Summary Metrics - {benchmark_name}", fontsize=13)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{benchmark_name}_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
