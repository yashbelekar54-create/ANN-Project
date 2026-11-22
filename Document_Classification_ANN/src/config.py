"""Configuration helpers shared by training, evaluation and the Streamlit app."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: Dict[str, Any], key: str) -> Path:
    """Return the absolute path for a key of ``cfg['paths']``."""
    return PROJECT_ROOT / cfg["paths"][key]


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories used by the project."""
    for key in ("raw_dir", "processed_dir", "models_dir", "figures_dir", "metrics_dir", "reports_dir"):
        resolve(cfg, key).mkdir(parents=True, exist_ok=True)
