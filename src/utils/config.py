from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib
import json
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def dict_hash(d: Dict[str, Any]) -> str:
    # stable hash for cache versioning (ignore ordering)
    j = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:12]
