"""Player provenance uses the shared tracker without ball model references."""

from training.common.provenance import ROOT
from training.common.tracking import log_metrics as log_metrics
from training.common.tracking import tracked_run as _tracked_run


def tracked_run(config, manifest):
    paths = list((ROOT / "training/players").rglob("*.py"))
    paths += list((ROOT / "training/common").glob("*.py"))
    return _tracked_run(config, manifest, root=ROOT, source_paths=paths)
