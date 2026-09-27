ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help
.PHONY: help setup-court calibrate-court test-court
help:
	@echo 'make setup-court'
	@echo 'make calibrate-court VIDEO=... CHECKPOINT=... OUTPUT=... [COURT_DEVICE=cuda:0]'
	@echo 'make test-court'
setup-court calibrate-court test-court:
	$(MAKE) -C "$(ROOT_DIR)" $@
