ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT_DIR)/.venv/bin/python
DVC ?= $(ROOT_DIR)/.venv/bin/dvc

.PHONY: help annotate-ball export-ball data-configure-r2 data-status data-track data-push data-pull

help:
	@echo 'make annotate-ball VIDEO="/path/to/clip.mp4"'

annotate-ball:
	@test -n "$(VIDEO)" || { echo 'Error: the VIDEO variable is required.' >&2; echo 'Usage: make annotate-ball VIDEO="/path/to/clip.mp4"' >&2; exit 2; }
	cd "$(ROOT_DIR)" && "$(PYTHON)" -m training.ball.annotator "$(ROOT_DIR)$(VIDEO)"


export-ball:
	@echo 'Training export is not implemented yet. The annotator already saves the .ballann.json file.' >&2
	@exit 2
