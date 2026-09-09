"""Train NBA's TrackNetV5 + TOTNet ideas: triplet augmentation and weighted WBCE."""

from training.ball.learning.config import parse_config, preflight
from training.ball.learning.tracking import tracked_run


def train(config):
    manifest = preflight(config)
    from models_factory import build_model

    from training.ball.learning.torch_data import SDKDataset
    from training.ball.learning.torch_loop import fit, seed_everything
    from training.ball.learning.totnet import (
        TripletAugmentation,
        VisibilityWeightedTrackNetLoss,
    )

    device = seed_everything(config)
    train_data = SDKDataset(config, "train", TripletAugmentation(config))
    val_data = SDKDataset(config, "val")
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
    config["loss"] = {
        "type": "VisibilityWeightedTrackNetLoss",
        "ball_present_weight": config["geometry"]["ball_present_weight"],
        "reduction": "mean",
        "epsilon": 1e-6,
    }
    with tracked_run(config, manifest) as tracking:
        model = build_model(config["architecture"])
        loss = VisibilityWeightedTrackNetLoss(
            occluded_weight=config["geometry"]["ball_present_weight"]
        )
        return fit(model, train_data, val_data, loss, config, device, tracking)


if __name__ == "__main__":
    train(parse_config("tracknet_v5_totnet"))
