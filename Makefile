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
	@echo 'make test-players  # Player contracts and shared infrastructure (CPU)'
	@echo 'make test-players-mlflow PYTHON=/path/to/python  # Local MLflow integration'
	@echo 'make annotate-players MEDIA=... [PLAYERS_ROOT=datas]'
	@echo 'make preannotate-players MEDIA=... CHECKPOINT=... | CACHE=...'
	@echo 'make export-players | verify-players-export  # Paired YOLO/COCO exports'
	@echo 'make train-players-yolo | train-players-rfdetr CONFIG=... [RESUME=...]'
	@echo 'make evaluate-players CONFIG=... | compare-players RUNS="... ..." REPORT_OUTPUT=...'
	@echo 'make pipeline-players | predict-players | register-players CONFIG=...'

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

.PHONY: test-players test-players-mlflow
test-players:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m unittest discover -s training/players/tests -v

test-players-mlflow:
	cd "$(ROOT_DIR)" && PLAYERS_MLFLOW_TESTS=1 "$(PYTHON)" -m unittest training.players.tests.test_tracking -v

MEDIA ?= $(VIDEO)
PLAYERS_ROOT ?= datas
PLAYER_STEP ?= 30
PLAYER_START ?= 0
PLAYER_STOP ?=
PLAYER_CLASS ?= 0
PLAYER_DEVICE ?= cpu
MATCH_ID ?=
VENUE_ID ?=
PLAYER_SPLIT ?=
CACHE ?=
CACHE_VIDEO_SHA256 ?=
PLAYERS_ANNOTATION_ARGS = "$(MEDIA)" --root "$(PLAYERS_ROOT)" --step "$(PLAYER_STEP)" --start "$(PLAYER_START)" $(if $(PLAYER_STOP),--stop "$(PLAYER_STOP)",) $(if $(MATCH_ID),--match-id "$(MATCH_ID)",) $(if $(VENUE_ID),--venue-id "$(VENUE_ID)",) $(if $(PLAYER_SPLIT),--split "$(PLAYER_SPLIT)",)
.PHONY: annotate-players preannotate-players
annotate-players:
	@test -n "$(MEDIA)" || { echo 'MEDIA (or VIDEO) is required' >&2; exit 2; }
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.players.annotator $(PLAYERS_ANNOTATION_ARGS)

preannotate-players:
	@test -n "$(MEDIA)" || { echo 'MEDIA (or VIDEO) is required' >&2; exit 2; }
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.players.annotator.preannotate $(PLAYERS_ANNOTATION_ARGS) --player-class "$(PLAYER_CLASS)" --device "$(PLAYER_DEVICE)" $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) $(if $(CACHE),--cache "$(CACHE)",) $(if $(CACHE_VIDEO_SHA256),--cache-video-sha256 "$(CACHE_VIDEO_SHA256)",)

PLAYERS_OUTPUT ?= exports/players
PLAYERS_EXPORT_CONFIG ?= training/players/configs/players_export.yaml
.PHONY: export-players verify-players-export
export-players:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.players.export --source "$(PLAYERS_ROOT)" --output "$(PLAYERS_OUTPUT)" --config "$(PLAYERS_EXPORT_CONFIG)"

verify-players-export:
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.players.export --output "$(PLAYERS_OUTPUT)" --verify

# Player model dependencies live in a separate environment from ball and annotation.
PLAYERS_PYTHON ?= $(ROOT_DIR)/.venv-players/bin/python
.PHONY: train-players-yolo train-players-rfdetr test-players-training
train-players-yolo:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.train_yolo --config "$(if $(CONFIG),$(CONFIG),training/players/configs/train_yolo_quick.yaml)" $(if $(RESUME),--resume "$(RESUME)",)

train-players-rfdetr:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.train_rfdetr --config "$(if $(CONFIG),$(CONFIG),training/players/configs/train_rfdetr_quick.yaml)" $(if $(RESUME),--resume "$(RESUME)",)

test-players-training:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m unittest training.players.tests.test_training_integration -v

.PHONY: evaluate-players compare-players test-players-evaluation
evaluate-players:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.evaluation evaluate --config "$(if $(CONFIG),$(CONFIG),training/players/configs/evaluate_yolo.yaml)" $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",)

# RUNS is a space-separated list of local evaluation run directories.
compare-players:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.evaluation compare $(RUNS) --output "$(if $(REPORT_OUTPUT),$(REPORT_OUTPUT),runs/players/comparison)" $(if $(INCLUDE_SMOKE),--include-smoke,)

test-players-evaluation:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m unittest training.players.tests.test_evaluation -v

.PHONY: pipeline-players predict-players register-players test-players-pipeline
pipeline-players:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.orchestration.pipeline --config "$(CONFIG)" $(if $(RESTART_INCOMPLETE),--restart-incomplete,)

predict-players:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.predict $(if $(CONFIG),--config "$(CONFIG)",) $(if $(VIDEO),--video "$(VIDEO)",) $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",)

register-players:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m training.players.registry --config "$(CONFIG)"

test-players-pipeline:
	cd "$(ROOT_DIR)" && "$(PLAYERS_PYTHON)" -m unittest training.players.tests.test_pipeline -v
