ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help setup-jersey predict-jersey test-jersey
help:
	@echo 'See README.md for model setup, temporal budgets and CPU/GPU inference.'
	@echo 'make setup-jersey [JERSEY_SOURCE=/local/parseq.pt]'
	@echo 'make predict-jersey VIDEO=... CHECKPOINT=... OUTPUT=...'
	@echo 'make test-jersey PYTHON=/path/to/python'
setup-jersey predict-jersey test-jersey:
	$(MAKE) -C "$(ROOT_DIR)" $@
