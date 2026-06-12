from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "figures": root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
