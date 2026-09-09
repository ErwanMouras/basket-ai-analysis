"""Train the reference TensorFlow TrackNetV4 on memory-mapped V4 exports."""

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from training.ball.learning.config import parse_config, preflight, resume_contract
from training.ball.learning.data import V4Arrays
from training.ball.learning.metrics import metric_values, update_counts
from training.ball.learning.tracking import log_metrics, tracked_run


def wbce(targets, predictions):
    """Reference V4 custom_loss, averaged over batch, frames and pixels."""
    import tensorflow as tf

    epsilon = tf.keras.backend.epsilon()
    return tf.reduce_mean(
        -(
            tf.square(1 - predictions)
            * targets
            * tf.math.log(tf.clip_by_value(predictions, epsilon, 1))
            + tf.square(predictions)
            * (1 - targets)
            * tf.math.log(tf.clip_by_value(1 - predictions, epsilon, 1))
        )
    )


def configure_tensorflow(config):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(config["cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    devices = tf.config.list_physical_devices("GPU")
    requested = config["device"]
    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif requested.startswith("cuda:"):
        index = int(requested.split(":")[1])
        if index >= len(devices):
            raise ValueError(
                "Requested TensorFlow GPU is unavailable; use cpu or install tensorflow[and-cuda]"
            )
        tf.config.set_visible_devices(devices[index], "GPU")
        tf.config.experimental.set_memory_growth(devices[index], True)
    else:
        raise ValueError("TensorFlow device must be cpu or cuda:N")
    tf.keras.utils.set_random_seed(config["seed"])
    if config["deterministic"]:
        tf.config.experimental.enable_op_determinism()


def run_epoch(model, optimizer, dataset, config, epoch, training):
    import tensorflow as tf

    total_loss, count, counts = 0.0, 0, Counter()
    limit = config["max_train_batches" if training else "max_val_batches"]
    for index, (images, targets) in enumerate(dataset.batches(epoch)):
        if limit is not None and index >= limit:
            break
        with tf.GradientTape() as tape:
            predictions = model(images, training=training)
            loss = wbce(targets, predictions)
            if model.losses:
                loss += tf.add_n(model.losses)
        tf.debugging.assert_all_finite(loss, "Non-finite V4 loss")
        if training:
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, config["grad_clip"])
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, "Non-finite V4 gradient")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            update_counts(
                counts,
                predictions.numpy(),
                targets,
                config["threshold"],
                config["tolerance_px"],
            )
        total_loss += float(loss) * len(images)
        count += len(images)
    if not count:
        raise ValueError("No V4 batches were produced")
    metrics = {"loss": total_loss / count, "samples": count}
    if not training:
        metrics.update(metric_values(counts))
    prefix = "train" if training else "val"
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def save_checkpoint(tf_checkpoint, directory, epoch, best, config, run_id):
    directory.mkdir(exist_ok=True)
    # Each directory is a complete model + optimizer checkpoint, including optimizer iterations.
    tf_checkpoint.write(str(directory / "state"))
    (directory / "training.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "best_val_loss": best,
                "contract": resume_contract(config),
                "run_id": run_id,
            },
            indent=2,
        )
    )


def train(config):
    manifest = preflight(config)
    configure_tensorflow(config)
    import tensorflow as tf
    from models.TrackNetV4 import TrackNetV4

    train_data, val_data = V4Arrays(config, "train"), V4Arrays(config, "val")
    config["architecture"] = {
        "type": "TrackNetV4",
        "fusion": config["fusion"],
        "input_height": config["geometry"]["input_height"],
        "input_width": config["geometry"]["input_width"],
    }
    config["loss"] = {"type": "custom_loss", "reduction": "mean", "epsilon": 1e-7}
    with tracked_run(config, manifest) as tracking:
        client, run_id, output = tracking
        model = TrackNetV4(
            config["geometry"]["input_height"],
            config["geometry"]["input_width"],
            config["fusion"],
        )
        optimizer_type = {
            "Adam": tf.keras.optimizers.Adam,
            "AdamW": tf.keras.optimizers.AdamW,
            "SGD": tf.keras.optimizers.SGD,
            "Adadelta": tf.keras.optimizers.Adadelta,
        }[config["optimizer"]]
        optimizer = optimizer_type(
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        optimizer.build(model.trainable_variables)
        runtime = output / "optimizer.resolved.json"
        runtime.write_text(json.dumps(optimizer.get_config(), indent=2))
        client.log_artifact(run_id, str(runtime), "provenance")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        start_epoch, best = 0, float("inf")
        if config["resume"]:
            directory = Path(config["resume"])
            state = json.loads((directory / "training.json").read_text())
            if state["contract"] != resume_contract(config):
                raise ValueError(
                    "Resume configuration/data differs from the V4 checkpoint"
                )
            checkpoint.read(str(directory / "state")).assert_consumed()
            start_epoch, best = state["epoch"] + 1, state["best_val_loss"]
            shutil.copytree(directory.parent / "best", output / "best")
            client.set_tag(run_id, "resumed_from_run", state["run_id"])
        if start_epoch >= config["epochs"]:
            raise ValueError("Checkpoint already reached the configured epoch budget")
        client.log_param(run_id, "parameter_count", model.count_params())
        try:
            for epoch in range(start_epoch, config["epochs"]):
                start = time.monotonic()
                rate = config["learning_rate"]
                if config["scheduler"] == "cosine":
                    rate = (
                        config["min_learning_rate"]
                        + (rate - config["min_learning_rate"])
                        * (1 + np.cos(np.pi * epoch / config["epochs"]))
                        / 2
                    )
                optimizer.learning_rate.assign(rate)
                metrics = run_epoch(model, optimizer, train_data, config, epoch, True)
                metrics.update(
                    run_epoch(model, optimizer, val_data, config, epoch, False)
                )
                improved = metrics["val/loss"] < best
                best = min(best, metrics["val/loss"])
                save_checkpoint(
                    checkpoint, output / "last", epoch, best, config, run_id
                )
                if improved:
                    save_checkpoint(
                        checkpoint, output / "best", epoch, best, config, run_id
                    )
                metrics.update(
                    learning_rate=rate, epoch_seconds=time.monotonic() - start
                )
                log_metrics(client, run_id, metrics, epoch)
                print(
                    f"Epoch {epoch + 1}: train={metrics['train/loss']:.6f}, val={metrics['val/loss']:.6f}",
                    flush=True,
                )
        finally:
            for name in ("best", "last"):
                if (output / name).exists():
                    client.log_artifacts(
                        run_id, str(output / name), f"checkpoints/{name}"
                    )
        return output


if __name__ == "__main__":
    train(parse_config("tracknet_v4"))
