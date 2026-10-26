"""Two experimental perception paths sharing the existing downstream box contract."""
from pathlib import Path
import numpy as np
from training.common.provenance import file_hash
from training.players.evaluation.metrics import iou
from training.segmentation.masks import encode


class SegmentedPeople:
    def __init__(self, people, config, *, detector_factory=None, model_factory=None):
        self.config, self.people_config = config, people
        path = Path(config["weights"]).resolve(strict=True)
        if detector_factory is None:
            from pipeline.adapters import People
            detector_factory = People
        self.detector = detector_factory(people)
        if model_factory is None:
            from ultralytics import SAM, YOLO
            model_factory = SAM if config["strategy"] == "box_prompt" else lambda p: YOLO(p, task="segment")
        self.model = model_factory(str(path))
        self.roles = dict(self.detector.roles)
        if config["strategy"] == "direct":
            # This detector is used for referees only. Its player boxes never reach output.
            self.roles = {k: v for k, v in self.detector.roles.items() if v == "referee"}
            if not self.roles:
                raise ValueError("Direct segmentation requires a referee detector")
            for cls, role in config["class_roles"].items():
                name = self.model.names.get(int(cls))
                if name is None:
                    raise ValueError(f"Segmentation class {cls} is missing")
                if role == "referee":
                    raise ValueError("Direct path keeps referees in detection only")
                if name.lower() == "person" and role == "player" and config["person_role_policy"] != "non_referee_person_proxy":
                    raise ValueError("Generic person requires an explicit experimental player proxy")
                if name.lower() in ("player", "referee") and role not in (name.lower(), "unknown"):
                    raise ValueError("Segmentation class/role mismatch")
                self.roles[f"seg:{cls}"] = role
        self.provenance = {
            "experimental": True, "strategy": config["strategy"], "config": config,
            "detector": self.detector.provenance, "checkpoint_sha256": file_hash(path),
            "weights": str(path), "device": people["device"], "precision": "fp32",
            "box_policy": "preserve_detector_box" if config["strategy"] == "box_prompt" else "instance_model_box",
            "mask_policy": "binary_source_pixels_clipped_to_box_roi",
            "role_limitation": "non-referee person is only a player proxy; spectators and missed referees remain" if config["person_role_policy"] == "non_referee_person_proxy" else None,
        }

    def predict(self, image, *, masks=True, confidence=None):
        cfg, pcfg = self.config, self.people_config
        confidence = pcfg["confidence"] if confidence is None else confidence
        if cfg["strategy"] == "box_prompt":
            rows = self.detector.predict(image, confidence=confidence)
            if not masks:
                return rows
            indices = [i for i, d in enumerate(rows) if cfg["segment_referees"] or d["role"] != "referee"]
            if not indices:
                return rows  # Never invoke SAM's automatic segment-everything path on empty input.
            result = self.model.predict(image, bboxes=[rows[i]["bbox"] for i in indices],
                imgsz=cfg["resolution"], device=pcfg["device"], conf=0.0, half=False, verbose=False)[0]
            if result.masks is None or len(result.masks.data) != len(indices):
                raise RuntimeError("SAM did not return exactly one mask per prompt")
            # SAM's class column carries prompt indices, not semantic classes.
            prompts = [int(x) for x in result.boxes.cls.cpu().tolist()]
            if sorted(prompts) != list(range(len(indices))):
                raise RuntimeError("Invalid SAM prompt-to-mask mapping")
            for prompt, mask, score in zip(prompts, result.masks.data.cpu().numpy(), result.boxes.conf.cpu().tolist(), strict=True):
                row = rows[indices[prompt]]
                self._mask(row, mask, image.shape[:2], float(score))
            return rows
        # Request referee classes only: no preceding player detection in this path.
        referees = self.detector.predict(image, roles={"referee"}, confidence=confidence)
        result = self.model.predict(image, imgsz=cfg["resolution"], conf=confidence,
            classes=[int(k) for k in cfg["class_roles"]], device=pcfg["device"],
            max_det=pcfg["max_detections"], retina_masks=masks, half=False, verbose=False)[0]
        if len(result.boxes) and (result.masks is None or len(result.masks.data) != len(result.boxes)):
            raise RuntimeError("Instance boxes and masks do not align")
        rows = []
        if result.masks is not None:
            for box, score, cls, mask in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(), result.masks.data.cpu().numpy() if masks else [None] * len(result.boxes), strict=True):
                if any(r["confidence"] >= pcfg["confidence"] and iou(box, r["bbox"]) >= cfg["referee_iou"] for r in referees):
                    continue
                cls = int(cls)
                row = {"bbox": box, "confidence": float(score), "class_id": 0, "source_class_id": cls,
                    "role": cfg["class_roles"][str(cls)], "role_source": cfg["person_role_policy"]}
                if masks:
                    self._mask(row, mask, image.shape[:2], None)
                rows.append(row)
        return rows + referees

    @staticmethod
    def _mask(row, mask, shape, score):
        if mask.shape != shape:
            raise RuntimeError("Expected a binary mask in source image coordinates")
        if not np.isfinite(mask).all():
            raise RuntimeError("Nonfinite segmentation mask")
        payload = encode(mask > 0, row["bbox"])
        # Empty masks retain their boxes but cannot silently supply a valid OCR mask.
        row.update(segmentation={"status": "ok" if sum(payload["counts"][1::2]) else "empty",
                                 "quality_score": score, "mask": payload})
