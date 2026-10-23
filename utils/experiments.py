"""Seed control and self-contained records for final-model experiments."""

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4
import zipfile

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    """Reset each method/run, so CLI method order does not change its seed."""
    if deterministic:
        # Set before creating a CUDA context. Unsupported deterministic kernels
        # should raise rather than silently weakening the requested guarantee.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None  # Unmeasured cells in accuracy matrices, not JSON NaN.
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2,
                                    allow_nan=False) + "\n")


def create_experiment_dir(output_dir, benchmark):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_dir) / f"{benchmark}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_info(device):
    packages = {dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "device": str(device),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "packages": dict(sorted(packages.items()))}


def snapshot_source(repo_root, destination):
    """Archive the code actually run, including local uncommitted changes."""
    repo_root = Path(repo_root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo_root), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None

    # Explicit source roots exclude downloaded data and prior run artifacts.
    paths = [repo_root / "main.py", repo_root / "README.md"]
    paths += list(repo_root.glob("requirements*.txt"))
    for folder in ("models", "trainers", "benchmarks", "utils", "tests"):
        paths += sorted((repo_root / folder).rglob("*.py"))
    hashes = {}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                relative = path.relative_to(repo_root).as_posix()
                content = path.read_bytes()
                hashes[relative] = hashlib.sha256(content).hexdigest()
                archive.writestr(relative, content)
    return {"commit": commit, "working_tree_status": status,
            "dirty": bool(status) if status is not None else None,
            "source_sha256": hashes}


class _Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, value):
        self.terminal.write(value)
        self.file.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


@contextlib.contextmanager
def capture_log(path):
    with open(path, "w") as file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, file)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, file)):
            yield


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    return value


def aggregate_runs(records):
    """Sample standard deviation is undefined for a single run (JSON null)."""
    result = {}
    for method in sorted({record["method"] for record in records}):
        runs = [record for record in records if record["method"] == method]
        summary = {"seeds": [record["seed"] for record in runs], "n": len(runs)}
        for metric in ("average_accuracy", "backward_transfer", "forgetting_measure"):
            values = [record["metrics"][metric] for record in runs]
            summary[metric] = {"mean": statistics.mean(values),
                               "std": statistics.stdev(values) if len(values) > 1 else None}
        if all(record["metrics"]["task_free"] is not None for record in runs):
            summary["task_free"] = {}
            for metric in ("accuracy", "task_selection_accuracy"):
                values = [record["metrics"]["task_free"][metric] for record in runs]
                summary["task_free"][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None}
        result[method] = summary
    return result
