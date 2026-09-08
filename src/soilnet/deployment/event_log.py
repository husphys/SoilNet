from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, correlation_id: str, **fields: Any) -> None:
        row = {
            "event": event,
            "correlation_id": correlation_id,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.perf_counter_ns(),
            **fields,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
