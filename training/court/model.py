"""Adapter for the explicitly installed basketball_analysis YOLOv8x pose model."""

from pathlib import Path

import numpy as np

from training.common.provenance import file_hash

MODEL_URL = "https://drive.usercontent.google.com/download?id=1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0&export=download&confirm=t"
MODEL_SHA256 = "f6263105e5c2338fafcfd5a6fefd7d1d441e87364635e918dfdbb849f2df1377"
UPSTREAM = "https://github.com/abdullahtarek/basketball_analysis/tree/197eef8ad7dc0b4daeeaa2086275aeaca8fbb8eb"


class CourtKeypoints:
    def __init__(self, config):
        path = Path(config["weights"])
        if not path.is_file():
            raise FileNotFoundError("Court model missing: run make setup-court first")
        digest = file_hash(path)
        if digest != MODEL_SHA256:
            raise ValueError("Unsupported court weights: checkpoint and 18-point schema must match")
        from ultralytics import YOLO, __version__
        self.model = YOLO(str(path), task="pose")
        if list(self.model.model.kpt_shape) != [18, 3]:
            raise ValueError("Expected 18 court keypoints with confidence")
        self.config = config
        self.provenance = {"model": "basketball_analysis_yolov8x_pose18",
                           "checkpoint_sha256": digest, "source": UPSTREAM,
                           "ultralytics_version": __version__, "device": config["device"],
                           "resolution": config["resolution"]}

    def predict(self, image):
        result = self.model.predict(image, imgsz=self.config["resolution"],
                                    device=self.config["device"], conf=0.25,
                                    max_det=1, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return np.zeros((18, 3), dtype=np.float64)
        return result.keypoints.data[0].cpu().numpy().astype(np.float64)
