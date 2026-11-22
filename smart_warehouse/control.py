from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_order_surge_command(path: Path, zone_id: str = "C", count: int = 20) -> str:
    if zone_id not in {"A", "B", "C", "D"}:
        raise ValueError("zone_id must be A, B, C, or D.")
    if count <= 0:
        raise ValueError("count must be positive.")
    command_id = str(uuid.uuid4())
    payload = {
        "command_id": command_id,
        "action": "order_surge",
        "zone_id": zone_id,
        "count": int(count),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)
    return command_id


def read_command(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
