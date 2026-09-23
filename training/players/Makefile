ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help test-players test-players-mlflow

help:
	@echo 'Players: annotation, exports, training and common evaluation'
	@echo 'make test-players  # CPU contracts; no model or GPU required'
	@echo 'make test-players-mlflow PYTHON=/path/to/python  # Local MLflow integration'
	@echo 'See README.md, TRAINING.md and EVALUATION.md.'
	@echo 'make annotate-players MEDIA=... [PLAYERS_ROOT=datas]'
	@echo 'make preannotate-players MEDIA=... CHECKPOINT=... | CACHE=...'
	@echo 'make export-players | verify-players-export'
	@echo 'make train-players-yolo | train-players-rfdetr CONFIG=... [RESUME=...]'
	@echo 'make test-players-training  # Opt-in short real training integrations'
	@echo 'make evaluate-players CONFIG=... | compare-players RUNS="... ..." REPORT_OUTPUT=...'
	@echo 'make test-players-evaluation  # Analytical COCO fixtures'

test-players test-players-mlflow:
	$(MAKE) -C "$(ROOT_DIR)" $@

.PHONY: annotate-players preannotate-players
annotate-players preannotate-players:
	$(MAKE) -C "$(ROOT_DIR)" $@

.PHONY: export-players verify-players-export
export-players verify-players-export:
	$(MAKE) -C "$(ROOT_DIR)" $@

.PHONY: train-players-yolo train-players-rfdetr test-players-training
train-players-yolo train-players-rfdetr test-players-training:
	$(MAKE) -C "$(ROOT_DIR)" $@

.PHONY: evaluate-players compare-players test-players-evaluation
evaluate-players compare-players test-players-evaluation:
	$(MAKE) -C "$(ROOT_DIR)" $@
