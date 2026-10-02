"""Atomic, resumable checkpoints with explicit experiment provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 2
DEFAULT_SOURCE_FILES = ("train.py", "baseline.py", "model.py", "synthesis.py", "losses.py",
                        "checkpointing.py", "data.py", "preprocessing.py", "validation_bank.py")


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def capture_rng(shuffle_generator=None):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch": torch.get_rng_state().clone(),
        "shuffle": shuffle_generator.get_state().clone() if shuffle_generator is not None else None,
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state().cpu().clone()
    return state


def restore_rng(state, shuffle_generator=None):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["kind"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if shuffle_generator is not None:
        shuffle_generator.set_state(state["shuffle"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_files(names=None):
    directory = Path(__file__).resolve().parent
    names = DEFAULT_SOURCE_FILES if names is None else names
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names if (directory / name).exists()}


def collect_provenance(config, manifest, bank_digest=None, extra_sources=()):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("torch", "numpy", "Pillow", "scikit-learn", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "source_files": _source_files((*DEFAULT_SOURCE_FILES, *extra_sources)), "config_digest": config_digest(config),
        "split_digest": manifest["split_digest"], "bank_digest": bank_digest,
        "dataset_provenance": manifest.get("provenance", {}),
        "data_seed": manifest["config"]["seed"],
        "training_seed": config.get("training_seed", config.get("seed", 42)),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "reproducibility_note": "Exact CPU resume is tested in one fixed environment; accelerator kernels may be nondeterministic.",
    }


def prepare_output(output, resume=None):
    output = Path(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Run directory is nonempty; choose a new directory or explicitly resume last.pt")
    if resume is not None:
        resume = Path(resume).resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        if output.exists() and any(output.iterdir()) and resume.parent != output.resolve():
            raise FileExistsError("A nonempty run can only resume its own last.pt")
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_resume(checkpoint, config, split_digest, model_kind, bank_digest=None, actual_device=None):
    if checkpoint.get("schema_version") != SCHEMA_VERSION or not checkpoint.get("resumable"):
        raise ValueError("Expected an epoch-boundary schema-v2 last.pt resumable checkpoint")
    if checkpoint.get("model_kind") != model_kind or checkpoint.get("split_digest") != split_digest:
        raise ValueError("Resume model kind or audited split differs from this run")
    if checkpoint.get("config_digest") != config_digest(config):
        raise ValueError("Resume configuration differs; epochs, seeds, device and all training settings must match")
    if checkpoint.get("bank_digest") != bank_digest:
        raise ValueError("Resume challenge bank differs")
    if checkpoint["provenance"]["source_files"] != _source_files(checkpoint["provenance"]["source_files"]):
        raise ValueError("Training source changed since checkpoint; create a new declared run")
    packages = checkpoint["provenance"]["packages"]
    for name, version in packages.items():
        if version is not None and importlib.metadata.version(name) != version:
            raise ValueError(f"Resume package version changed: {name}")
    if checkpoint["provenance"]["cpu_threads"] != torch.get_num_threads():
        raise ValueError("Resume CPU thread count differs")
    if checkpoint["provenance"]["python"] != sys.version:
        raise ValueError("Resume Python version differs")
    if checkpoint["provenance"]["deterministic_algorithms"] != torch.are_deterministic_algorithms_enabled():
        raise ValueError("Resume deterministic algorithm setting differs")
    if actual_device is not None and checkpoint["device"] != str(actual_device):
        raise ValueError("Resume resolved device differs")
