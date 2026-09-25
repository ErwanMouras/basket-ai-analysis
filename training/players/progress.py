"""Player-only atomic progress events, optionally forwarded to an orchestrator."""

import os
import time
from pathlib import Path

from training.common.files import write_json

_started = time.monotonic()
_state = {}


def emit(**values):
    _state.update(values)
    _state.update(elapsed_seconds=time.monotonic() - _started, updated_at=time.time())
    _state.setdefault("eta_seconds", None)
    path = os.environ.get("PLAYERS_PROGRESS_PATH")
    if path:
        write_json(Path(path), _state)
    return dict(_state)
