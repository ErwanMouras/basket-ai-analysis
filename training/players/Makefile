ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help test-players test-players-mlflow

help:
	@echo 'Phase 1: player data contracts and shared infrastructure'
	@echo 'make test-players  # CPU contracts; no model or GPU required'
	@echo 'make test-players-mlflow PYTHON=/path/to/python  # Local MLflow integration'
	@echo 'See README.md for the next phases.'

test-players test-players-mlflow:
	$(MAKE) -C "$(ROOT_DIR)" $@
