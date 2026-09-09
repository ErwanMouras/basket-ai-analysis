"""Train the exact NBA SDK TrackNetV5, without TOTNet augmentations."""

from training.ball.learning.config import parse_config, preflight
from training.ball.learning.tracking import tracked_run


def train(config):
    manifest = preflight(config)
    from losses_factory.losses.tracknetv2_loss import TrackNetV2Loss
    from models_factory import build_model

    from training.ball.learning.torch_data import SDKDataset
    from training.ball.learning.torch_loop import fit, seed_everything

    device = seed_everything(config)
    train_data, val_data = SDKDataset(config, "train"), SDKDataset(config, "val")
    config["architecture"] = {
        "type": "TrackNetV5",
        "backbone": {"type": "TrackNetV2Backbone", "in_channels": 13},
        "neck": {"type": "TrackNetV2Neck"},
        "head": {
            "type": "R_STRHead",
            "in_channels": 64,
            "out_channels": 3,
            "img_size": (
                config["geometry"]["input_height"],
                config["geometry"]["input_width"],
            ),
            "patch_size": 16,
            "embed_dim": 256,
            "num_transformer_layers": 4,
            "num_transformer_heads": 2,
            "IsDraft": False,
            "dropout": True,
        },
    }
    config["loss"] = {"type": "TrackNetV2Loss", "reduction": "mean", "epsilon": 1e-6}
    with tracked_run(config, manifest) as tracking:
        model = build_model(config["architecture"])
        return fit(
            model, train_data, val_data, TrackNetV2Loss(), config, device, tracking
        )


if __name__ == "__main__":
    train(parse_config("tracknet_v5"))
