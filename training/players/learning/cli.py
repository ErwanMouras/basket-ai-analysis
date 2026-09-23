"""Shared entry point for explicit training and resume modes."""

import argparse

from training.players.learning.config import load_config
from training.players.learning.run import train


def main(family):
    parser = argparse.ArgumentParser(
        description=f"Train {family} player detection using verified train/val exports"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", help="Strict resume from a complete player checkpoint"
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Technical interruption after a saved epoch, for resume tests",
    )
    args = parser.parse_args()
    print(
        train(
            load_config(family, args.config, resume=args.resume),
            stop_after_epoch=args.stop_after_epoch,
        )
    )
