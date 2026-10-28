"""Pinned neural models with CUDA preprocessing, decoding and mask encoding."""

from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.cuda.core import letterbox, resize, tensor
from pipeline.cuda.graphs import CapturedCall
from pipeline.cuda.onnx_pose import Pose
from training.common.provenance import file_hash
from training.court.model import MODEL_SHA256 as COURT_SHA256
from training.jersey.config import MODEL_SHA256 as JERSEY_SHA256


class Yolo:
    def __init__(self, config, *, task="detect", digest=None):
        from ultralytics import YOLO

        self.config = config
        self.device = config["device"]
        path = Path(config["weights"]).resolve(strict=True)
        actual = file_hash(path)
        if digest and digest != actual:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        self.model = (
            YOLO(str(path), task=task)
            .model.to(device=self.device, dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        self.names = self.model.names
        self.nc = len(self.names)
        self.captured = None
        self.provenance = {
            "weights": str(path),
            "checkpoint_sha256": actual,
            "device": self.device,
            "backend": "native_CUDA_Torch",
            "preprocessing": "CUDA_letterbox_RGB",
            "decoding": "CUDA_NMS",
            "precision": "fp32",
        }

    def predict(
        self,
        image,
        *,
        classes=None,
        size=None,
        confidence=None,
        max_det=None,
        stretch=None,
    ):
        from ultralytics.utils.nms import non_max_suppression

        cfg = self.config
        if stretch:
            height, width = stretch
            batch = resize(image, height, width)[None] / 255
            gain = tensor(
                [image.shape[2] / width, image.shape[1] / height] * 2,
                device=self.device,
                dtype=torch.float32,
            )
            ratio, left, top = None, 0, 0
        else:
            prepared, ratio, left, top = letterbox(
                image, size or cfg.get("resolution", 640)
            )
            batch = prepared[None] / 255
        result = self.captured(batch) if self.captured else self.model(batch)
        prediction = non_max_suppression(
            result,
            conf_thres=cfg.get("confidence", 0.25)
            if confidence is None
            else confidence,
            iou_thres=cfg.get("iou", 0.7),
            classes=classes,
            nc=self.nc,
            max_det=max_det or cfg.get("max_detections", 100),
            end2end=getattr(self.model, "end2end", False),
        )[0]
        if not len(prediction):
            return prediction
        if stretch:
            prediction[:, :4] *= gain
        else:
            prediction[:, [0, 2]] = (prediction[:, [0, 2]] - left) / ratio
            prediction[:, [1, 3]] = (prediction[:, [1, 3]] - top) / ratio
            if prediction.shape[1] > 6:
                keypoints = prediction[:, 6:].view(len(prediction), -1, 3)
                keypoints[..., 0] = (keypoints[..., 0] - left) / ratio
                keypoints[..., 1] = (keypoints[..., 1] - top) / ratio
        prediction[:, [0, 2]] = prediction[:, [0, 2]].clamp(0, image.shape[2])
        prediction[:, [1, 3]] = prediction[:, [1, 3]].clamp(0, image.shape[1])
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite CUDA YOLO output")
        return prediction[
            (prediction[:, 2] > prediction[:, 0])
            & (prediction[:, 3] > prediction[:, 1])
        ]


class Segmentation:
    def __init__(self, config, device):
        from ultralytics.models.sam.predict import SAM2Predictor

        self.device = device
        self.config = config
        Path(config["weights"]).resolve(strict=True)
        # Construct parameters and eval caches on CUDA as well.
        with torch.device(device):
            self.predictor = SAM2Predictor(
                overrides={
                    "model": config["weights"],
                    "device": device,
                    "imgsz": 1024,
                    "half": False,
                    "verbose": False,
                }
            )
            self.predictor.setup_model(verbose=False)
        self.predictor._bb_feat_sizes = [[256, 256], [128, 128], [64, 64]]
        self.captured = None
        self.provenance = {
            "checkpoint_sha256": file_hash(Path(config["weights"])),
            "weights": config["weights"],
            "device": device,
            "backend": "SAM2_native_CUDA",
            "strategy": "box_prompt",
            "mask_encoding": "CUDA_roi_RLE",
        }

    def predict(self, image, boxes):
        if not len(boxes):
            return torch.empty(
                (0, *image.shape[-2:]), device=self.device, dtype=torch.bool
            ), torch.empty(0, device=self.device)
        prepared, ratio, _, _ = letterbox(image, 1024, center=False, square=True)
        normalized = ((prepared - self.predictor.mean) / self.predictor.std)[None]
        features = (
            self.captured(normalized)
            if self.captured
            else self.predictor.get_im_features(normalized)
        )
        prompts = (boxes * ratio).reshape(-1, 2, 2)
        labels = tensor([[2, 3]], device=self.device, dtype=torch.int32).expand(
            len(boxes), -1
        )
        logits, confidence = self.predictor._inference_features(
            features, points=prompts, labels=labels
        )
        logits = F.interpolate(
            logits[:, None], size=(1024, 1024), mode="bilinear", align_corners=False
        )
        h, w = round(image.shape[1] * ratio), round(image.shape[2] * ratio)
        masks = (
            F.interpolate(
                logits[:, :, :h, :w],
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            > 0
        )
        return masks, confidence


def encode_mask(mask, box):
    roi = torch.cat((box[:2].floor(), box[2:].ceil())).long()
    roi[[0, 2]] = roi[[0, 2]].clamp(0, mask.shape[1])
    roi[[1, 3]] = roi[[1, 3]].clamp(0, mask.shape[0])
    a, b, c, d = roi.tolist()  # allocation/slicing indices, not host arithmetic
    values = mask[b:d, a:c].flatten()
    if not len(values):
        raise ValueError("Empty CUDA mask ROI")
    changes = torch.nonzero(values[1:] != values[:-1]).flatten() + 1
    edges = torch.cat(
        (
            torch.zeros(1, device=mask.device, dtype=torch.long),
            changes,
            tensor([len(values)], device=str(mask.device), dtype=torch.long),
        )
    )
    counts = edges.diff()
    if bool(values[0]):
        counts = torch.cat(
            (torch.zeros(1, device=mask.device, dtype=torch.long), counts)
        )
    return {"encoding": "roi_rle_c", "roi": roi, "counts": counts}, bool(values.any())


class Jersey:
    def __init__(self, config):
        from training.jersey.vendor.parseq.model import PARSeq

        self.config = config
        self.device = config["device"]
        if file_hash(Path(config["weights"])) != JERSEY_SHA256:
            raise ValueError("PARSeq checkpoint mismatch")
        with torch.device(self.device):
            self.model = PARSeq(
                97, 25, (32, 128), (4, 8), 384, 6, 4, 12, 12, 4, 1, True, 1, 0.1
            )
        self.model.to(self.device)
        state = torch.load(
            config["weights"], map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.captured = None
        self.provenance = {
            "checkpoint_sha256": JERSEY_SHA256,
            "device": self.device,
            "backend": "PARSeq_native_CUDA",
            "decoder": "digits_only_v4_CUDA",
            "cpu_fallback": None,
        }

    def read(self, crops):
        if not self.captured:
            return self._read_eager(crops)
        tokens, confidence = [], []
        size = self.config["batch_size"]
        for start in range(0, len(crops), size):
            batch = crops[start : start + size]
            padded = F.pad(batch, (0, 0, 0, 0, 0, 0, 0, size - len(batch)))
            result, scores = self.captured(padded)
            tokens.append(result[: len(batch)])
            confidence.append(scores[: len(batch)])
        if not tokens:
            return self._read_eager(crops)
        return torch.cat(tokens), torch.cat(confidence)

    def _read_eager(self, crops):
        if not len(crops):
            return torch.empty(
                (0, 3), device=self.device, dtype=torch.long
            ), torch.empty(0, device=self.device)
        all_tokens, all_confidence = [], []
        for start in range(0, len(crops), self.config["batch_size"]):
            batch = crops[start : start + self.config["batch_size"]].float() / 127.5 - 1
            memory = self.model.encode(batch)
            steps = 3
            queries = self.model.pos_queries[:, :steps].expand(len(batch), -1, -1)
            mask = torch.ones(
                (steps, steps), dtype=torch.bool, device=self.device
            ).triu(1)
            context = torch.full(
                (len(batch), steps), 96, dtype=torch.long, device=self.device
            )
            context[:, 0] = 95
            tokens, probabilities = [], []
            for index in range(steps):
                length = index + 1
                decoded = self.model.decode(
                    context[:, :length],
                    memory,
                    mask[:length, :length],
                    tgt_query=queries[:, index:length],
                    tgt_query_mask=mask[index:length, :length],
                )
                logits = self.model.head(decoded).squeeze(1)
                allowed = torch.full_like(logits, -torch.inf)
                if index < 2:
                    allowed[:, 1:11] = logits[:, 1:11]
                if index > 0:
                    allowed[:, 0] = logits[:, 0]
                token = allowed.argmax(-1)
                tokens.append(token)
                probabilities.append(
                    logits.softmax(-1).gather(1, token[:, None]).squeeze(1)
                )
                if length < steps:
                    context[:, length] = token
            token = torch.stack(tokens, 1)
            probability = torch.stack(probabilities, 1)
            end = torch.where(token[:, 1] == 0, 1, 2)
            valid = (
                (token[:, 0] >= 1)
                & (token[:, 0] <= 10)
                & (token.gather(1, end[:, None])[:, 0] == 0)
            )
            confidence = torch.where(
                torch.arange(3, device=self.device)[None] <= end[:, None],
                probability,
                1,
            ).prod(1)
            all_tokens.append(token)
            all_confidence.append(torch.where(valid, confidence, 0))
        return torch.cat(all_tokens), torch.cat(all_confidence)


class Models:
    def __init__(self, config):
        self.people = Yolo(config["people"])
        for key, role in config["people"]["class_roles"].items():
            if self.people.names[int(key)].lower() != role:
                raise ValueError(
                    "CUDA detector requires explicit player/referee classes"
                )
        self.ball = Yolo(config["ball"], digest=config["ball"].get("checkpoint_sha256"))
        self.court = Yolo(config["court"], task="pose", digest=COURT_SHA256)
        self.pose = Pose(config["pose"])
        self.segmentation = Segmentation(
            config["segmentation"], config["people"]["device"]
        )
        self.jersey = Jersey(config["jersey"])
        self.provenance = {
            name: getattr(self, name).provenance
            for name in ("people", "ball", "court", "pose", "segmentation", "jersey")
        }

    def prepare_graphs(self, shape):
        # Kornia initializes dtype-size metadata with tensor factories at import.
        # Import once inside the caller's guarded CUDA device scope.
        __import__("kornia.color")
        height, width = shape
        for name in ("people", "court", "ball"):
            wrapper = getattr(self, name)
            if name == "ball":
                h, w = wrapper.config["input_height"], wrapper.config["input_width"]
            else:
                size = wrapper.config.get("resolution", 640)
                ratio = min(size / height, size / width)
                h, w = (
                    ((round(height * ratio) + 31) // 32) * 32,
                    ((round(width * ratio) + 31) // 32) * 32,
                )

            def forward(batch, model=wrapper.model):
                output = model(batch)
                return output[0] if isinstance(output, tuple) else output

            wrapper.captured = CapturedCall(
                forward, torch.zeros((1, 3, h, w), device=wrapper.device)
            )
            wrapper.provenance["cuda_graph"] = True
            wrapper.provenance["cuda_graph_details"] = wrapper.captured.metadata
        self.segmentation.captured = CapturedCall(
            self.segmentation.predictor.get_im_features,
            torch.zeros((1, 3, 1024, 1024), device=self.people.device),
        )
        self.segmentation.provenance["encoder_cuda_graph"] = True
        self.segmentation.provenance["cuda_graph_details"] = (
            self.segmentation.captured.metadata
        )
        self.jersey.captured = CapturedCall(
            self.jersey._read_eager,
            torch.zeros(
                (self.jersey.config["batch_size"], 3, 32, 128),
                device=self.people.device,
            ),
        )
        self.jersey.provenance["cuda_graph"] = True
        self.jersey.provenance["cuda_graph_details"] = self.jersey.captured.metadata
        self.pose.prepare_graphs()
