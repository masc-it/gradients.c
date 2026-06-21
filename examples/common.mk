# Shared build plumbing for examples.
#
# Usage from examples/<name>/Makefile:
#   ROOT := ../..
#   include ../common.mk
#
# Optional overrides:
#   BUILD_DIR       Root-relative or absolute core build directory. Defaults to
#                   build-<example-directory-name>.
#   CORE_CFLAGS     Passed to the root Makefile as CFLAGS for the core library.
#   CORE_OBJCFLAGS  Passed to the root Makefile as OBJCFLAGS for the core library.
#
# Reusable example helpers exposed to Makefiles:
#   GD_PROGRESS_SRC, GD_PROGRESS_HDR, GD_PROGRESS_DEPS
#   GD_EXAMPLE_CONFIG_SRC, GD_EXAMPLE_CONFIG_HDR, GD_EXAMPLE_CONFIG_DEPS

ifndef ROOT
$(error ROOT must be set before including ../common.mk)
endif

EXAMPLE_NAME ?= $(notdir $(CURDIR))
BUILD_DIR ?= build-$(EXAMPLE_NAME)

ifneq ($(filter /%,$(BUILD_DIR)),)
BUILD_PATH := $(BUILD_DIR)
else
BUILD_PATH := $(ROOT)/$(BUILD_DIR)
endif

LIB ?= $(BUILD_PATH)/libgradients.a
METALLIB ?= $(BUILD_PATH)/gradients.metallib

EXAMPLE_COMMON_DIR := $(ROOT)/examples/common
GD_PROGRESS_SRC := $(EXAMPLE_COMMON_DIR)/gd_progress.c
GD_PROGRESS_HDR := $(EXAMPLE_COMMON_DIR)/gd_progress.h
GD_PROGRESS_DEPS := $(GD_PROGRESS_SRC) $(GD_PROGRESS_HDR)
GD_EXAMPLE_CONFIG_SRC := $(EXAMPLE_COMMON_DIR)/gd_example_config.c
GD_EXAMPLE_CONFIG_HDR := $(EXAMPLE_COMMON_DIR)/gd_example_config.h
GD_EXAMPLE_CONFIG_DEPS := $(GD_EXAMPLE_CONFIG_SRC) $(GD_EXAMPLE_CONFIG_HDR)
CPPFLAGS += -I$(EXAMPLE_COMMON_DIR)

CORE_BUILD_ARGS = BUILD_DIR='$(BUILD_DIR)'
ifneq ($(origin CORE_CFLAGS),undefined)
CORE_BUILD_ARGS += CFLAGS='$(CORE_CFLAGS)'
endif
ifneq ($(origin CORE_OBJCFLAGS),undefined)
CORE_BUILD_ARGS += OBJCFLAGS='$(CORE_OBJCFLAGS)'
endif

.PHONY: build-lib tools

build-lib:
	$(MAKE) -C $(ROOT) $(CORE_BUILD_ARGS) build

tools:
	$(MAKE) -C $(ROOT) $(CORE_BUILD_ARGS) tools

$(LIB) $(METALLIB): build-lib ;
