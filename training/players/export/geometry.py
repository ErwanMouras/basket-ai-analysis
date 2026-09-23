"""Invertible source-to-export box geometry using the actual rounded image size."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    source_width: int
    source_height: int
    width: int
    height: int
    content_width: int
    content_height: int
    scale_x: float
    scale_y: float
    left: int
    top: int

    @classmethod
    def build(cls, width, height, config):
        out_w, out_h = config.resize_width or width, config.resize_height or height
        cw, ch = out_w, out_h
        if config.resize_mode == "letterbox":
            scale = min(out_w / width, out_h / height)
            cw = max(1, min(out_w, round(width * scale)))
            ch = max(1, min(out_h, round(height * scale)))
        return cls(
            width,
            height,
            out_w,
            out_h,
            cw,
            ch,
            cw / width,
            ch / height,
            (out_w - cw) // 2,
            (out_h - ch) // 2,
        )

    def bbox(self, box, *, inverse=False):
        x1, y1, x2, y2 = box
        if inverse:
            return [
                (x1 - self.left) / self.scale_x,
                (y1 - self.top) / self.scale_y,
                (x2 - self.left) / self.scale_x,
                (y2 - self.top) / self.scale_y,
            ]
        return [
            x1 * self.scale_x + self.left,
            y1 * self.scale_y + self.top,
            x2 * self.scale_x + self.left,
            y2 * self.scale_y + self.top,
        ]

    def image(self, image):
        if image.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("Decoded frame dimensions do not match source")
        if (self.content_width, self.content_height) != (
            self.source_width,
            self.source_height,
        ):
            interpolation = (
                cv2.INTER_AREA
                if max(self.scale_x, self.scale_y) < 1
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(
                image,
                (self.content_width, self.content_height),
                interpolation=interpolation,
            )
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[
            self.top : self.top + self.content_height,
            self.left : self.left + self.content_width,
        ] = image
        return canvas

    def to_dict(self):
        return asdict(self)
