"""Train Ultralytics ball detection with one explicit MLflow run."""

import shutil
from pathlib import Path

import yaml

from training.ball.learning.config import parse_config, preflight, resume_contract
from training.ball.learning.data import frame_index
from training.ball.learning.tracking import tracked_run


def train(config):
    manifest = preflight(config)
    import torch
    import ultralytics

    from training.ball.learning.yolo_trainer import BallDetectionTrainer

    if ultralytics.__version__ != "8.4.39":
        raise ValueError(
            "This checkpoint/callback adapter requires ultralytics==8.4.39"
        )
    if config["device"] != "cpu" and not config["device"].startswith("cuda:"):
        raise ValueError("YOLO supports a single cpu or cuda:N device")
    if config["device"].startswith("cuda:") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    root = Path(config["dataset"])
    for split in ("train", "val"):
        frame_index(root, split)
    torch.set_num_threads(config["cpu_threads"])
    previous = None
    if config["resume"]:
        previous = torch.load(config["resume"], map_location="cpu", weights_only=False)
        state = previous.get("ball_training")
        if not state or state["contract"] != resume_contract(config):
            raise ValueError("Resume requires a matching complete ball YOLO checkpoint")
        if previous["epoch"] + 1 >= config["epochs"]:
            raise ValueError("Checkpoint already reached the configured epoch budget")
    with tracked_run(config, manifest) as tracking:
        client, run_id, output = tracking
        data = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "ball"},
        }
        data_path = output / "yolo_data.yaml"
        data_path.write_text(yaml.safe_dump(data))
        overrides = dict(
            config["yolo"],
            model=config["resume"] or config["weights"],
            data=str(data_path),
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["input_width"],
            optimizer=config["optimizer"],
            lr0=config["learning_rate"],
            weight_decay=config["weight_decay"],
            seed=config["seed"],
            device=config["device"].replace("cuda:", ""),
            workers=config["workers"],
            amp=config["precision"] == "amp",
            deterministic=config["deterministic"],
            cache=False,
            val=True,
            split="val",
            save=True,
            save_period=-1,
            plots=False,
            project=str(output),
            name="yolo",
            exist_ok=False,
            resume=config["resume"] or False,
        )
        trainer = BallDetectionTrainer(config, tracking, overrides)
        if previous:
            shutil.copy2(Path(config["resume"]).parent / "best.pt", trainer.best)
            client.set_tag(
                run_id, "resumed_from_run", previous["ball_training"]["run_id"]
            )
        try:
            trainer.train()
        finally:
            for path in (trainer.best, trainer.last):
                if path.is_file():
                    client.log_artifact(run_id, str(path), "checkpoints")
            if trainer.csv.is_file():
                client.log_artifact(run_id, str(trainer.csv))
        if not trainer.best.is_file() or not trainer.last.is_file():
            raise RuntimeError("YOLO finished without best/last checkpoints")
        return output


if __name__ == "__main__":
    train(parse_config("yolo"))
