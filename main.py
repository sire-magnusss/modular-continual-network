"""
Continual Learning Architecture — Main Experiment Runner
==========================================================

Usage:
    python main.py                          # default: all methods on Split-CIFAR-10
    python main.py --benchmark mnist        # use Permuted MNIST instead
    python main.py --methods naive ewc      # run only specific methods
    python main.py --epochs 10              # more epochs per task
    python main.py --tasks 3               # only 3 tasks (faster for testing)

What you'll see:
    After training on each task, we evaluate all previously seen tasks.
    Watch the Naive method forget Task 0 once Task 1 training starts.
    Compare EWC and PackNet's ability to retain old knowledge.

Output:
    - Console: accuracy matrix, metrics
    - results/plots/: PNG plots comparing all methods
"""

import argparse
import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import torch
from utils.device import get_device
from utils.metrics import MetricTracker
from utils.visualization import plot_results

from benchmarks import SplitCIFAR10, PermutedMNIST, SplitCIFAR100
from models.backbone import CIFARBackbone, MNISTBackbone
from models.ewc import EWCModel
from models.packnet import PackNetModel
from models.mcn import MCN
from models.mcn_ablations import MCNNoRouter, MCNNoGate, MCNBaseOnly
from trainers.naive_trainer import NaiveTrainer
from trainers.ewc_trainer import EWCTrainer
from trainers.packnet_trainer import PackNetTrainer
from trainers.mcn_trainer import MCNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Baselines")
    parser.add_argument("--benchmark", choices=["cifar10", "cifar100", "mnist"],
                        default="cifar10",
                        help="Which benchmark to run (default: cifar10)")
    parser.add_argument("--methods", nargs="+",
                        choices=["naive", "ewc", "packnet", "mcn",
                                 "mcn_no_router", "mcn_no_gate", "mcn_base_only"],
                        default=["naive", "ewc", "packnet", "mcn"],
                        help="Which methods to run (default: all)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per task per method (default: 5)")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Override number of tasks (default: benchmark default)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--ewc-lambda", type=float, default=5000.0,
                        help="EWC regularization strength (default: 5000)")
    parser.add_argument("--packnet-prune", type=float, default=0.5,
                        help="PackNet prune fraction per task (default: 0.5)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for dataset downloads (default: ./data)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    return parser.parse_args()


def build_benchmark(args):
    if args.benchmark == "cifar10":
        benchmark = SplitCIFAR10(data_dir=args.data_dir, batch_size=args.batch_size)
        if args.tasks:
            benchmark.num_tasks = min(args.tasks, 5)
            benchmark.name = f"SplitCIFAR10_{benchmark.num_tasks}tasks"
        return benchmark, "cifar10"

    elif args.benchmark == "cifar100":
        benchmark = SplitCIFAR100(data_dir=args.data_dir, batch_size=args.batch_size)
        if args.tasks:
            benchmark.num_tasks = min(args.tasks, 20)
            benchmark.name = f"SplitCIFAR100_{benchmark.num_tasks}tasks"
        return benchmark, "cifar100"

    else:  # mnist
        num_tasks = args.tasks or 5
        benchmark = PermutedMNIST(
            num_tasks=num_tasks, data_dir=args.data_dir, batch_size=args.batch_size
        )
        return benchmark, "mnist"


def build_model(benchmark_type: str, num_tasks: int, method: str,
                ewc_lambda: float, packnet_prune: float):
    in_ch = 1 if benchmark_type == "mnist" else 3
    sz    = 28 if benchmark_type == "mnist" else 32
    n_cls = {"cifar10": 2, "cifar100": 5, "mnist": 10}[benchmark_type]

    # CIFAR (diverse object classes): full base freeze wins
    # MNIST (permuted, same structure): adaptive high-level freeze helps
    adaptive = (benchmark_type == "mnist")

    if method == "mcn":
        return MCN(num_tasks=num_tasks, num_classes_per_task=n_cls,
                   in_channels=in_ch, input_size=sz, freeze_all=not adaptive)
    elif method == "mcn_no_router":
        return MCNNoRouter(num_tasks=num_tasks, num_classes_per_task=n_cls,
                           in_channels=in_ch, input_size=sz, freeze_all=not adaptive)
    elif method == "mcn_no_gate":
        return MCNNoGate(num_tasks=num_tasks, num_classes_per_task=n_cls,
                         in_channels=in_ch, input_size=sz, freeze_all=not adaptive)
    elif method == "mcn_base_only":
        return MCNBaseOnly(num_tasks=num_tasks, num_classes_per_task=n_cls,
                           base_dim=512, in_channels=in_ch, input_size=sz)

    if benchmark_type == "mnist":
        backbone = MNISTBackbone(num_tasks=num_tasks, num_classes_per_task=n_cls)
    else:  # cifar10 or cifar100
        backbone = CIFARBackbone(num_tasks=num_tasks, num_classes_per_task=n_cls)

    if method == "naive":
        return backbone
    elif method == "ewc":
        return EWCModel(backbone, ewc_lambda=ewc_lambda)
    elif method == "packnet":
        return PackNetModel(backbone, prune_fraction=packnet_prune)


def run_method(method_name: str, model, benchmark,
               device: torch.device, args) -> MetricTracker:
    tracker = MetricTracker(benchmark.num_tasks)

    if method_name == "naive":
        trainer = NaiveTrainer(model, device, lr=args.lr,
                               epochs_per_task=args.epochs)
    elif method_name == "ewc":
        trainer = EWCTrainer(model, device, lr=args.lr,
                             epochs_per_task=args.epochs)
    elif method_name == "packnet":
        trainer = PackNetTrainer(model, device, lr=args.lr,
                                 epochs_phase1=args.epochs,
                                 epochs_phase2=max(1, args.epochs // 3))
    elif method_name in ("mcn", "mcn_no_router", "mcn_no_gate", "mcn_base_only"):
        trainer = MCNTrainer(model, device, lr=args.lr,
                             epochs_per_task=args.epochs)

    trainer.run(benchmark, tracker)
    return tracker


def print_final_comparison(results: dict):
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Method':<12} {'Avg Acc':>10} {'BWT':>10} {'Forgetting':>12}")
    print("-"*50)
    for method, tracker in results.items():
        s = tracker.summary()
        print(f"{method:<12} "
              f"{s['average_accuracy']*100:>9.1f}%"
              f"{s['backward_transfer']*100:>9.1f}%"
              f"{s['forgetting_measure']*100:>11.1f}%")
    print()
    print("Avg Acc    — higher is better (accuracy after all tasks)")
    print("BWT        — less negative is better (negative = forgetting)")
    print("Forgetting — lower is better (drop from peak accuracy)")


def main():
    args = parse_args()
    device = get_device()

    print(f"\nBenchmark : {args.benchmark.upper()}")
    print(f"Methods   : {', '.join(args.methods)}")
    print(f"Epochs    : {args.epochs} per task")
    print(f"Device    : {device}")

    benchmark, benchmark_type = build_benchmark(args)
    print(f"Tasks     : {benchmark.num_tasks}")

    results = {}

    for method_name in args.methods:
        model = build_model(
            benchmark_type=benchmark_type,
            num_tasks=benchmark.num_tasks,
            method=method_name,
            ewc_lambda=args.ewc_lambda,
            packnet_prune=args.packnet_prune
        )
        tracker = run_method(method_name, model, benchmark, device, args)
        results[method_name.capitalize()] = tracker

    print_final_comparison(results)

    if not args.no_plots and len(results) > 0:
        print("\nGenerating plots...")
        plot_results(results, benchmark.name)


if __name__ == "__main__":
    main()
