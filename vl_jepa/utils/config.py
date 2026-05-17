from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and merge optional key/value overrides."""
    config = OmegaConf.load(Path(path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(dict(overrides)))
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value
