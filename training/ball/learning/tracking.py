"""Ball compatibility entry points for shared experiment tracking."""

from training.common import provenance
from training.common.tracking import flatten as flatten
from training.common.tracking import log_metrics as log_metrics
from training.common.tracking import tracked_run as _tracked_run

from .references import ROOT


def git_state():
    return provenance.git_state(ROOT)


def _source_paths():
    files = list((ROOT / "training/ball/learning").glob("*.py"))
    files += list((ROOT / "training/ball").glob("train_*.py"))
    files += list((ROOT / "training/ball/evaluation").glob("*.py"))
    files += list((ROOT / "training/common").glob("*.py"))
    return files


def source_fingerprints():
    return provenance.source_fingerprints(ROOT, _source_paths())


def tracked_run(config, manifest):
    return _tracked_run(
        config, manifest, root=ROOT, source_paths=_source_paths(),
        reference_paths=[ROOT / "training/ball/configs/references.json"],
    )
