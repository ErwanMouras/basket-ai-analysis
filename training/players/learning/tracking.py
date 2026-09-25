"""Player provenance uses the shared tracker without ball model references."""

from training.common.provenance import ROOT
from training.common.tracking import log_metrics as log_metrics
from training.common.tracking import tracked_run as _tracked_run
from contextlib import contextmanager
from training.players.progress import emit


@contextmanager
def tracked_run(config, manifest):
    paths = list((ROOT / "training/players").rglob("*.py"))
    paths += list((ROOT / "training/common").glob("*.py"))
    with _tracked_run(config, manifest, root=ROOT, source_paths=paths) as tracking:
        emit(run_id=tracking[1], output=str(tracking[2]),
             epochs_total=config.get("epochs"), epochs_completed=0, batch=None)
        yield tracking
