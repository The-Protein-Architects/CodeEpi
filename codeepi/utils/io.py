"""Small file / config helpers used across the CodeEpi CLIs."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Union

import yaml


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping at top level of {p}, got {type(data)}")
    return data


def save_yaml(path: PathLike, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``path`` as UTF-8 YAML (no unicode escapes)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def merge_paths(cfg: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    merged = dict(paths)
    merged.update(cfg.get("paths", {}))
    out["paths"] = merged
    return out


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
