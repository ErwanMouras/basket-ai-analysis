ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help test-players test-players-mlflow

help:
	@echo 'Players: contracts, annotation and preannotation'
	@echo 'make test-players  # CPU contracts; no model or GPU required'
	@echo 'make test-players-mlflow PYTHON=/path/to/python  # Local MLflow integration'
	@echo 'See README.md for the next phases.'
	@echo 'make annotate-players MEDIA=... [PLAYERS_ROOT=datas]'
	@echo 'make preannotate-players MEDIA=... CHECKPOINT=... | CACHE=...'

test-players test-players-mlflow:
	$(MAKE) -C "$(ROOT_DIR)" $@

.PHONY: annotate-players preannotate-players
annotate-players preannotate-players:
	$(MAKE) -C "$(ROOT_DIR)" $@
