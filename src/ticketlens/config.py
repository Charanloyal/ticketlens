"""Configuration loader and settings manager for TicketLens."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import yaml

if TYPE_CHECKING:
    import torch


def get_project_root() -> Path:
    """Return the root path of the TicketLens project."""
    # Assuming this file is at src/ticketlens/config.py
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_device() -> Union["torch.device", str]:
    """Detect and return CUDA device if available, otherwise CPU.

    torch is imported lazily so the lightweight serving image (no torch installed)
    can still start; in that case the plain string "cpu" is returned.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with default fallback."""
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root
    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    cfg["paths"][k] = str(root / p)

    return cfg
