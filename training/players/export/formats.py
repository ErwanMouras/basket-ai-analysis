"""Deterministic projections of the shared frame index into YOLO and COCO."""

from pathlib import PurePosixPath

from training.common.provenance import object_hash


def frame_record(source, frame, geometry, image_format, image_sha256):
    frame_id = object_hash([source["source_id"], frame["frame_index"]])
    return {
        "frame_id": frame_id,
        "source_id": source["source_id"],
        "frame_index": frame["frame_index"],
        "review_status": "verified",
        "split": source["split"],
        "match_id": source["match_id"],
        "venue_id": source["venue_id"],
        "image": f"images/{source['split']}/{frame_id}.{image_format}",
        "image_sha256": image_sha256,
        "width": geometry.width,
        "height": geometry.height,
        "transform": geometry.to_dict(),
        "boxes": [
            {**box, "source_bbox": box["bbox"], "bbox": geometry.bbox(box["bbox"])}
            for box in sorted(frame["boxes"], key=lambda box: box["object_id"])
        ],
    }


def label_path(record):
    image = PurePosixPath(record["image"])
    return PurePosixPath("labels", *image.parts[1:]).with_suffix(".txt").as_posix()


def yolo_label(record):
    lines = []
    for box in record["boxes"]:
        x1, y1, x2, y2 = box["bbox"]
        width, height = record["width"], record["height"]
        values = (
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        lines.append("0 " + " ".join(format(v, ".17g") for v in values) + "\n")
    return "".join(lines)


def yolo_config(splits):
    return {
        "names": {0: "player"},
        "train": None,
        "val": None,
        **{split: f"images/{split}" for split in splits},
    }


def coco_document(records, split):
    images, annotations = [], []
    for record in records:
        if record["split"] != split:
            continue
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": record["image"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for box in record["boxes"]:
            x1, y1, x2, y2 = box["bbox"]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "player"}],
    }


def statistics(records, splits):
    def count(items):
        return {
            "images": len(items),
            "boxes": sum(len(r["boxes"]) for r in items),
            "negatives": sum(not r["boxes"] for r in items),
        }

    return {
        "total": count(records),
        "splits": {s: count([r for r in records if r["split"] == s]) for s in splits},
    }
