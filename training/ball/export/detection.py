"""YOLO boxes and COCO boxes/center keypoints for a single ball class."""

from pathlib import Path

import yaml

from .files import write_json


def write_yolo(root: Path, records: list[dict], splits):
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        image = Path(record["image"])
        label = root / "labels" / Path(*image.parts[1:]).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        text = ""
        if record["bbox"] is not None:
            x, y, width, height = record["bbox"]
            values = (
                (x + width / 2) / record["width"],
                (y + height / 2) / record["height"],
                width / record["width"],
                height / record["height"],
            )
            text = "0 " + " ".join(f"{value:.8f}" for value in values) + "\n"
        label.write_text(text, encoding="utf-8")
    # Omitting `path` makes current Ultralytics resolve paths from the YAML directory.
    data = {"train": None, "val": None}
    data.update({split: f"images/{split}" for split in splits})
    data["names"] = {0: "ball"}
    (root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    return {"frames": len(records), "boxes": sum(r["has_position"] for r in records)}


def write_coco(root: Path, records: list[dict], splits):
    categories = [{"id": 1, "name": "ball", "keypoints": ["center"], "skeleton": []}]
    for split in splits:
        images, annotations = [], []
        for image_id, record in enumerate(records, 1):
            if record["split"] != split:
                continue
            images.append(
                {
                    "id": image_id,
                    "file_name": record["image"],
                    "width": record["width"],
                    "height": record["height"],
                }
            )
            if record["bbox"] is not None:
                annotations.append(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": record["bbox"],
                        "area": record["bbox"][2] * record["bbox"][3],
                        "iscrowd": 0,
                        "keypoints": [
                            *record["position"],
                            1 if record["occluded"] else 2,
                        ],
                        "num_keypoints": 1,
                    }
                )
        write_json(
            root / "annotations" / f"{split}.json",
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
        )
    return {"frames": len(records), "boxes": sum(r["has_position"] for r in records)}
