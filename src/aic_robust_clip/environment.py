"""Read-only diagnostics and explicit, host-bound training enrollment."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .contracts import read_json, write_json
from .runtime import LOCAL_POLICY, RuntimeLimitError, RuntimePolicy

# Irreversible identifier only; no hostname or machine-id is published.
DEVELOPMENT_HOST = "d18bf1f8c8a9065559750bd412c310cf69f122a8e10be14035c8764462a1d077"


def machine_fingerprint():
    source = Path("/etc/machine-id")
    if not source.is_file():
        raise RuntimeLimitError("Linux/WSL machine-id is required for host binding")
    return hashlib.sha256((source.read_text().strip() + "\0" + platform.node()).encode()).hexdigest()


def machine_policy(path=None):
    if path is None:
        return LOCAL_POLICY
    value = read_json(path)
    host = machine_fingerprint()
    if value.get("role") != "training" or value.get("fingerprint") != host or host == DEVELOPMENT_HOST:
        raise RuntimeLimitError("training configuration is not bound to this permitted experiment host")
    return RuntimePolicy(machine="bound-4060-experiment", allow_formal=True)


def environment_report(root="."):
    report = {"python": sys.version, "platform": platform.platform(),
              "fingerprint": machine_fingerprint(), "cuda_available": False,
              "disk_free_bytes": shutil.disk_usage(root).free, "dependencies": {},
              "real_model_startup": "not_run", "formal_training": "not_run"}
    for package in ("torch", "transformers", "Pillow", "numpy", "huggingface-hub"):
        try:
            report["dependencies"][package] = version(package)
        except PackageNotFoundError:
            report["dependencies"][package] = None
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": props.name, "memory_bytes": props.total_memory}
    except ImportError:
        pass
    executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if Path(executable).is_file():
        report["nvidia_smi"] = subprocess.run([executable], capture_output=True, text=True, timeout=15).stdout
    return report


def bind_training(path):
    report = environment_report()
    if report["fingerprint"] == DEVELOPMENT_HOST:
        raise RuntimeLimitError("this development host cannot be enrolled for formal training")
    if not report["cuda_available"] or "4060" not in report.get("gpu", {}).get("name", ""):
        raise RuntimeLimitError("enrollment requires the separate CUDA-visible RTX 4060")
    if Path(path).exists():
        raise FileExistsError(path)
    write_json(path, {"role": "training", "fingerprint": report["fingerprint"], "preflight": report})
    return report
