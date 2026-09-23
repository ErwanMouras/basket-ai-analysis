"""Train RF-DETR players with explicit MLflow tracking."""

from training.players.learning.cli import main

if __name__ == "__main__":
    main("rfdetr")
