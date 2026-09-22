ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT_DIR)/.venv/bin/python
DVC ?= $(ROOT_DIR)/.venv/bin/dvc
BALL_SOURCE ?= datas
BALL_OUTPUT ?= exports/ball
BALL_EXPORT_CONFIG ?= training/ball/configs/ball_export.yaml
BALL_TRAINING_CONFIG ?= training/ball/configs/ball_training.yaml
TRACKNET_LAYOUTS ?=
BALL_EXPORT = cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.export --source "$(BALL_SOURCE)" --output "$(BALL_OUTPUT)" --config "$(BALL_EXPORT_CONFIG)" --training-config "$(BALL_TRAINING_CONFIG)" $(if $(TRACKNET_LAYOUTS),--tracknet-layouts "$(TRACKNET_LAYOUTS)",)

.PHONY: help annotate-ball export-ball export-ball-yolo export-ball-coco export-ball-tracknet-totnet test-ball

help:
	@echo 'make annotate-ball VIDEO="/path/to/clip.mp4"'
	@echo 'make export-ball  # Export all ball formats'
	@echo 'make export-ball-yolo | export-ball-coco | export-ball-tracknet-totnet'
	@echo 'make export-ball-tracknet-totnet TRACKNET_LAYOUTS=sdk,v3  # Optional reference layouts'
	@echo 'make test-ball  # Annotation, export and training contract tests'
	@echo 'make setup-ball-models  # Install pinned reference model sources'
	@echo 'make train-ball-yolo | train-ball-tracknet-v3 | train-ball-tracknet-v4'
	@echo 'make train-ball-tracknet-v5 | train-ball-tracknet-v5-totnet'
	@echo 'Training options: PYTHON=/path/to/python CONFIG=recipe.yaml RESUME=checkpoint'
	@echo 'make evaluate-ball CHECKPOINT=runs/ball/<run>/best.pt [CONFIG=evaluation.yaml]'

annotate-ball:
	@test -n "$(VIDEO)" || { echo 'Error: the VIDEO variable is required.' >&2; echo 'Usage: make annotate-ball VIDEO="/path/to/clip.mp4"' >&2; exit 2; }
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.annotator "$(VIDEO)"


export-ball:
	$(BALL_EXPORT) --format all

export-ball-yolo:
	$(BALL_EXPORT) --format yolo

export-ball-coco:
	$(BALL_EXPORT) --format coco

export-ball-tracknet-totnet:
	$(BALL_EXPORT) --format tracknet-totnet

test-ball:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m unittest discover -s training/ball/tests -v

# CONFIG selects an alternative recipe for any individual training command.
CONFIG ?=
RESUME ?=
BALL_TRAIN = cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.train_
BALL_TRAIN_ARGS = $(if $(CONFIG),--config "$(CONFIG)",) $(if $(RESUME),--resume "$(RESUME)",)
.PHONY: train-ball-yolo train-ball-tracknet-v3 train-ball-tracknet-v4 train-ball-tracknet-v5 train-ball-tracknet-v5-totnet setup-ball-models
train-ball-yolo:
	$(BALL_TRAIN)yolo $(BALL_TRAIN_ARGS)
train-ball-tracknet-v3:
	$(BALL_TRAIN)tracknet_v3 $(BALL_TRAIN_ARGS)
train-ball-tracknet-v4:
	$(BALL_TRAIN)tracknet_v4 $(BALL_TRAIN_ARGS)
train-ball-tracknet-v5:
	$(BALL_TRAIN)tracknet_v5 $(BALL_TRAIN_ARGS)
train-ball-tracknet-v5-totnet:
	$(BALL_TRAIN)tracknet_v5_totnet $(BALL_TRAIN_ARGS)
setup-ball-models:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.learning.references

CHECKPOINT ?=
.PHONY: evaluate-ball
evaluate-ball:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.evaluation $(if $(CONFIG),--config "$(CONFIG)",) $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",)
