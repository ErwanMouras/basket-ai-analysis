"""MLflow pyfunc flavor for a portable player checkpoint and its provenance."""

import json
from pathlib import Path

import numpy as np

from training.common.provenance import file_hash, object_hash
from training.players.inference import predict_image
from training.players.models import Detector


def detector_from_bundle(path, device="cpu"):
    path = Path(path)
    metadata = json.loads((path / "candidate.json").read_text())
    if object_hash({k: v for k, v in metadata.items() if k != "candidate_id"}) != metadata["candidate_id"]:
        raise ValueError("Registered candidate metadata integrity mismatch")
    weights = path / metadata["weights_file"]
    model = metadata["model"]
    if file_hash(weights) != model["checkpoint_sha256"]:
        raise ValueError("Registered checkpoint integrity mismatch")
    detector = Detector(model["family"], model["variant"], weights, device=device,
                        resolution=model["resolution"], source_class=model["source_class"])
    return detector, metadata


class PlayerModel:
    def __init__(self, path):
        self.detector, self.metadata = detector_from_bundle(path)

    def predict(self, model_input, params=None):
        if params:
            raise ValueError("Registry inference uses the recorded prediction protocol")
        if isinstance(model_input, np.ndarray) and model_input.ndim == 3:
            model_input = [model_input]
        return [predict_image(self.detector, image, self.metadata["protocol"]) for image in model_input]


def _load_pyfunc(path):
    return PlayerModel(path)
