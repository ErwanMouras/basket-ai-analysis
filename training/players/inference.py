"""The same batch-one FP32 detector path for evaluation, video and registry."""

from training.players.evaluation.metrics import normalize_predictions
from training.players.export.config import ExportConfig
from training.players.export.geometry import Geometry


def predict_image(detector, image, config, record=None):
    if record is None:
        height, width = image.shape[:2]
        record = {"transform": Geometry.build(width, height, ExportConfig()).to_dict()}
    return normalize_predictions(
        record,
        detector.predict(image, confidence=config["score_floor"],
                         max_detections=config["max_detections"], square=True),
        score_floor=config["score_floor"], max_detections=config["max_detections"],
    )
