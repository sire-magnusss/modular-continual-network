"""Command-line runner for continual-learning experiments."""

import argparse
import sys
import os
from pathlib import Path
import time
import traceback

# Allow direct execution from the repository root.
sys.path.insert(0, os.path.dirname(__file__))

import torch
from utils.device import get_device
from utils.metrics import MetricTracker
from utils.visualization import plot_results
from utils.experiments import (seed_everything, create_experiment_dir, environment_info,
                               snapshot_source, write_json, capture_log, cpu_state,
                               aggregate_runs)

from benchmarks import SplitCIFAR10, PermutedMNIST, SplitCIFAR100
from models.backbone import CIFARBackbone, MNISTBackbone
from models.ewc import EWCModel
from models.packnet import PackNetModel
from models.mcn import MCN
from models.mcn_v2 import MCNv2
from models.mcn_ablations import MCNNoRouter, MCNNoGate, MCNBaseOnly
from models.hat import HATModel, HATMNISTModel
from trainers.naive_trainer import NaiveTrainer
from trainers.ewc_trainer import EWCTrainer
from trainers.packnet_trainer import PackNetTrainer
from trainers.mcn_trainer import MCNTrainer
from trainers.hat_trainer import HATTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Baselines")
    parser.add_argument("--benchmark", choices=["cifar10", "cifar100", "mnist"],
                        default="cifar10",
                        help="Which benchmark to run (default: cifar10)")
    parser.add_argument("--methods", nargs="+",
                        choices=["naive", "ewc", "packnet", "hat", "mcn", "mcn_v2",
                                 "mcn_no_router", "mcn_no_gate", "mcn_base_only"],
                        default=["naive", "ewc", "packnet", "mcn"],
                        help="Methods to run (default: naive ewc packnet mcn)")
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Independent training seeds; each method is reset to each seed")
    parser.add_argument("--permutation-seed", type=int, default=42,
                        help="MNIST task-definition seed, separate from training seeds")
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic PyTorch algorithms; unsupported kernels raise")
    parser.add_argument("--task-free", action="store_true",
                        help="Also evaluate MCN entropy routing in the global label space")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional PyTorch CPU thread count")
    parser.add_argument("--output-dir", default="runs",
                        help="Root for unique experiment directories")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs, batch-size and finite lr must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if any(seed < 0 or seed >= 2**32 for seed in [*args.seeds, args.permutation_seed]):
        parser.error("seeds must be between 0 and 2**32 - 1")
    if args.tasks is not None:
        maximum = {"cifar10": 5, "cifar100": 20}.get(args.benchmark)
        if args.tasks < 1 or (maximum and args.tasks > maximum):
            parser.error("tasks must be positive and within the benchmark's task count")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    return args


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
            num_tasks=num_tasks, data_dir=args.data_dir, batch_size=args.batch_size,
            seed=args.permutation_seed
        )
        return benchmark, "mnist"


def build_model(benchmark_type: str, num_tasks: int, method: str,
                ewc_lambda: float, packnet_prune: float):
    in_ch = 1 if benchmark_type == "mnist" else 3
    sz    = 28 if benchmark_type == "mnist" else 32
    n_cls = {"cifar10": 2, "cifar100": 5, "mnist": 10}[benchmark_type]

    # Preserve the historical benchmark-specific policy. Its relative merits
    # require controlled reruns; this is not evidence that either policy wins.
    adaptive = (benchmark_type == "mnist")

    if method == "hat":
        if benchmark_type == "mnist":
            return HATMNISTModel(num_tasks=num_tasks,
                                 num_classes_per_task=n_cls)
        else:
            return HATModel(num_tasks=num_tasks,
                            num_classes_per_task=n_cls,
                            in_channels=in_ch, input_size=sz)

    if method == "mcn":
        return MCN(num_tasks=num_tasks, num_classes_per_task=n_cls,
                   in_channels=in_ch, input_size=sz, freeze_all=not adaptive)
    elif method == "mcn_v2":
        return MCNv2(num_tasks=num_tasks, num_classes_per_task=n_cls,
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
    elif method_name == "hat":
        trainer = HATTrainer(model, device, lr=args.lr,
                             epochs_per_task=args.epochs)
    elif method_name == "ewc":
        trainer = EWCTrainer(model, device, lr=args.lr,
                             epochs_per_task=args.epochs)
    elif method_name == "packnet":
        trainer = PackNetTrainer(model, device, lr=args.lr,
                                 epochs_phase1=args.epochs,
                                 epochs_phase2=max(1, args.epochs // 3))
    elif method_name in ("mcn", "mcn_v2", "mcn_no_router", "mcn_no_gate", "mcn_base_only"):
        trainer = MCNTrainer(model, device, lr=args.lr,
                             epochs_per_task=args.epochs, task_free=args.task_free)

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
              f"{s['backward_transfer']*100:>8.1f}pp"
              f"{s['forgetting_measure']*100:>10.1f}pp")
    print()
    print("Avg Acc    - higher is better (accuracy after all tasks)")
    print("BWT        - less negative is better (negative = forgetting)")
    print("Forgetting - lower is better (drop from peak accuracy)")


def main():
    args = parse_args()
    seed_everything(args.seeds[0], args.deterministic)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    print(f"\nBenchmark : {args.benchmark.upper()}")
    print(f"Methods   : {', '.join(args.methods)}")
    print(f"Epochs    : {args.epochs} per task")
    print(f"Device    : {device}")

    benchmark, benchmark_type = build_benchmark(args)
    print(f"Tasks     : {benchmark.num_tasks}")

    experiment_dir = create_experiment_dir(args.output_dir, args.benchmark)
    print(f"Artifacts : {experiment_dir.resolve()}")
    source = snapshot_source(Path(__file__).resolve().parent, experiment_dir / "source.zip")
    environment = environment_info(device)
    write_json(experiment_dir / "experiment.json", {
        "schema_version": 1, "arguments": vars(args), "source": source,
        "environment": environment, "protocol": benchmark.protocol(),
        "metric_units": "fractions; multiply by 100 for percent/percentage points",
        "primary_evaluation": "task-incremental (task ID supplied)",
    })
    records = []
    for seed in args.seeds:
        results = {}
        for method_name in args.methods:
            run_dir = experiment_dir / f"seed_{seed}" / method_name
            run_dir.mkdir(parents=True)
            model_config = {
                "benchmark_type": benchmark_type, "num_tasks": benchmark.num_tasks,
                "method": method_name, "ewc_lambda": args.ewc_lambda,
                "packnet_prune": args.packnet_prune,
            }
            record = {"schema_version": 1, "method": method_name, "seed": seed,
                      "status": "running", "arguments": vars(args),
                      "model_config": model_config,
                      "experiment_manifest": "../../experiment.json",
                      "source_archive": "../../source.zip"}
            write_json(run_dir / "run.json", record)
            with capture_log(run_dir / "train.log"):
                try:
                    seed_everything(seed, args.deterministic)
                    print(f"Method: {method_name}; seed: {seed}; device: {device}")
                    model = build_model(**model_config)
                    record["total_parameters"] = sum(p.numel() for p in model.parameters())
                    started = time.perf_counter()
                    tracker = run_method(method_name, model, benchmark, device, args)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elif device.type == "mps":
                        torch.mps.synchronize()
                    record["train_and_evaluation_seconds"] = time.perf_counter() - started
                    record["metrics"] = tracker.summary()
                    # Final inference checkpoint, not an optimizer/training-resume snapshot.
                    torch.save({"schema_version": 1, "model_config": model_config,
                                "state_dict": cpu_state(model.state_dict()), "seed": seed},
                               run_dir / "final_model.pt")
                    record["checkpoint_bytes"] = (run_dir / "final_model.pt").stat().st_size
                    record["status"] = "complete"
                    write_json(run_dir / "run.json", record)
                    records.append(record)
                    results[method_name.capitalize()] = tracker
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    write_json(run_dir / "run.json", record)
                    traceback.print_exc()
                    raise
            write_json(experiment_dir / "summary.json", aggregate_runs(records))
        print(f"\nSeed {seed}")
        print_final_comparison(results)
        if not args.no_plots:
            plot_results(results, benchmark.name,
                         output_dir=experiment_dir / f"seed_{seed}" / "plots")

    print("\nAcross-seed average accuracy (sample standard deviation):")
    for method, summary in aggregate_runs(records).items():
        metric = summary["average_accuracy"]
        spread = f"{metric['std'] * 100:.2f} pp" if metric['std'] is not None else "undefined (one seed)"
        print(f"  {method}: {metric['mean'] * 100:.2f}%; std={spread}; n={summary['n']}")
    print(f"Run records and summary: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
