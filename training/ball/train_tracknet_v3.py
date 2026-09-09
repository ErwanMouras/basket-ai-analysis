"""Train the reference V3 TrackNet localization branch with sample mixup."""

from training.ball.learning.config import parse_config, preflight
from training.ball.learning.tracking import tracked_run


def train(config):
    manifest = preflight(config)
    from model import TrackNet
    from utils.metric import WBCELoss

    from training.ball.learning.torch_data import V3Dataset
    from training.ball.learning.torch_loop import fit, seed_everything

    device = seed_everything(config)
    train_data, val_data = V3Dataset(config, "train"), V3Dataset(config, "val")
    config["architecture"] = {
        "type": "TrackNet",
        "in_dim": 3 * config["geometry"]["sequence_length"],
        "out_dim": config["geometry"]["sequence_length"],
    }
    config["loss"] = {"type": "WBCELoss", "reduction": "mean", "epsilon": 1e-7}
    with tracked_run(config, manifest) as tracking:
        length = config["geometry"]["sequence_length"]
        model = TrackNet(in_dim=3 * length, out_dim=length)
        return fit(
            model,
            train_data,
            val_data,
            WBCELoss,
            config,
            device,
            tracking,
            mixup_alpha=config["mixup_alpha"],
        )


if __name__ == "__main__":
    train(parse_config("tracknet_v3"))
