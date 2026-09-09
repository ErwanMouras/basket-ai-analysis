# Keep a single definition of the commands at the repository root.
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help annotate-ball export-ball export-ball-yolo export-ball-coco export-ball-tracknet-totnet test-ball
help annotate-ball export-ball export-ball-yolo export-ball-coco export-ball-tracknet-totnet test-ball:
	$(MAKE) -C "$(ROOT_DIR)" $@
