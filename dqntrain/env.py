"""Environment detection for env_report() and checkpoint meta.json."""
from __future__ import annotations

import importlib
import importlib.metadata
import platform


def _version(package):
    try:
        return importlib.metadata.version(package)
    except Exception:
        pass
    try:
        module = importlib.import_module(package)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def env_info():
    """Return a dict of detected versions; missing packages become None."""
    info = {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "cuda": None,
        "rocm": None,
        "unsloth": _version("unsloth"),
        "bitsandbytes": _version("bitsandbytes"),
        "trl": _version("trl"),
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "huggingface_hub": _version("huggingface_hub"),
        "peft": _version("peft"),
        "dqntrain": importlib.import_module("dqntrain").__version__,
    }
    try:
        import torch

        if getattr(torch.version, "hip", None):
            info["rocm"] = torch.version.hip
        elif getattr(torch.version, "cuda", None):
            info["cuda"] = torch.version.cuda
        elif getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
            info["mps"] = "available"
    except Exception:
        pass
    return info


def env_report():
    """Print detected versions. Warns about missing packages, never fails."""
    info = env_info()
    optional = {"unsloth", "bitsandbytes", "cuda", "rocm"}
    for key, value in info.items():
        if value is None and key not in optional:
            print(f"WARNING: {key} not found")
        else:
            print(f"{key}: {value}")
    return info
