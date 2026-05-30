# gradients.c Makefile
# Primary commands build first, then run their workload.
# Examples:
#   make test        # build lib + tests, then run tests
#   make mlp         # build lib + examples/mlp/mlp, then run MLP example
#   make check       # build, run docs checks, run tests

SHELL := /bin/bash

# ----- Project layout -------------------------------------------------------
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
EXAMPLE_DIR := examples
DOCS_DIR := docs

LIB_NAME := gradients
LIB := $(BUILD_DIR)/lib$(LIB_NAME).a

# ----- Toolchain ------------------------------------------------------------
CC ?= cc
AR ?= ar
RM := rm -rf

CPPFLAGS ?= -I$(INCLUDE_DIR)
CFLAGS ?= -std=c11 -O0 -g3 -Wall -Wextra -Wpedantic -Werror \
          -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
          -Wmissing-prototypes -Wno-unused-parameter
LDFLAGS ?=
LDLIBS ?= -lm

# Override from CLI when needed:
#   make test CFLAGS='-std=c11 -O2 -g -Wall'

# ----- Platform / Metal toolchain -------------------------------------------
UNAME_S := $(shell uname -s)
METAL_DIR := $(SRC_DIR)/backends/metal

# Metal builds only on macOS; default on there, off elsewhere. Override with
# `make GD_ENABLE_METAL=0` to force a CPU-only build on macOS.
ifeq ($(UNAME_S),Darwin)
GD_ENABLE_METAL ?= 1
else
GD_ENABLE_METAL ?= 0
endif

METALLIB :=

# ----- Source discovery -----------------------------------------------------
ifeq ($(GD_ENABLE_METAL),1)
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' 2>/dev/null | sort)
MSRC := $(shell find $(METAL_DIR) -type f -name '*.m' 2>/dev/null | sort)
METAL_SHADERS := $(shell find $(METAL_DIR) -type f -name '*.metal' 2>/dev/null | sort)
OBJCFLAGS := $(CPPFLAGS) $(CFLAGS) -fobjc-arc -x objective-c
LDLIBS += -framework Metal -framework Foundation -framework QuartzCore
# The .metal shader compiler ships with full Xcode, not the Command Line Tools.
# Compile shaders only when it is present; the Obj-C path still builds without it.
METAL_TOOL := $(shell xcrun --find metal 2>/dev/null)
ifneq ($(strip $(METAL_SHADERS)),)
ifneq ($(strip $(METAL_TOOL)),)
METALLIB := $(BUILD_DIR)/gradients.metallib
endif
endif
else
# Exclude the Metal backend entirely on non-macOS / disabled builds.
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' -not -path '$(METAL_DIR)/*' 2>/dev/null | sort)
MSRC :=
METAL_SHADERS :=
endif

MOBJ := $(patsubst %.m,$(BUILD_DIR)/%.o,$(MSRC))
OBJ := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC)) $(MOBJ)
DEP := $(OBJ:.o=.d)

TEST_SRC := $(shell find $(TEST_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRC))

EXAMPLE_SRC := $(shell find $(EXAMPLE_DIR) -type f -name '*.c' 2>/dev/null | sort)
EXAMPLE_BINS := $(patsubst %.c,$(BUILD_DIR)/%,$(EXAMPLE_SRC))

MLP_SRC := $(EXAMPLE_DIR)/mlp/mlp.c
MLP_BIN := $(BUILD_DIR)/$(EXAMPLE_DIR)/mlp/mlp

# ----- Public commands ------------------------------------------------------
.PHONY: help all build check test tests mlp examples docs-check clean list

help:
	@printf '%s\n' 'gradients.c commands:'
	@printf '%s\n' '  make build       build library'
	@printf '%s\n' '  make test        build library + tests, then run all tests'
	@printf '%s\n' '  make check       build, run docs checks, then run tests'
	@printf '%s\n' '  make mlp         build library + MLP example, then run it'
	@printf '%s\n' '  make examples    build library + all examples, then run all examples'
	@printf '%s\n' '  make docs-check  build, then validate docs links/references'
	@printf '%s\n' '  make list        show discovered source/test/example files'
	@printf '%s\n' '  make clean       remove build artifacts'
	@printf '%s\n' ''
	@printf '%s\n' 'Primary run commands intentionally build first.'

all: check

build: $(LIB) $(METALLIB)
	@printf '[build] ok (metal=%s)\n' '$(GD_ENABLE_METAL)'

check: build docs-check test
	@printf '[check] ok\n'

test tests: build $(TEST_BINS)
ifeq ($(strip $(TEST_BINS)),)
	@printf '[test] no tests found under %s/ yet\n' '$(TEST_DIR)'
else
	@set -euo pipefail; \
	for t in $(TEST_BINS); do \
		printf '[test] %s\n' "$$t"; \
		"$$t"; \
	done
endif
	@printf '[test] ok\n'

mlp: build
ifeq ($(wildcard $(MLP_SRC)),)
	@printf '[mlp] missing %s; skipped\n' '$(MLP_SRC)'
else
	@$(MAKE) --no-print-directory $(MLP_BIN)
	@printf '[mlp] run %s\n' '$(MLP_BIN)'
	@$(MLP_BIN)
endif

examples: build $(EXAMPLE_BINS)
ifeq ($(strip $(EXAMPLE_BINS)),)
	@printf '[examples] no examples found under %s/ yet\n' '$(EXAMPLE_DIR)'
else
	@set -euo pipefail; \
	for x in $(EXAMPLE_BINS); do \
		printf '[example] %s\n' "$$x"; \
		"$$x"; \
	done
endif
	@printf '[examples] ok\n'

docs-check: build
	@test -f $(DOCS_DIR)/design_spec.md
	@test -f $(DOCS_DIR)/plan_mlp.md
	@grep -q 'design_spec.md' $(DOCS_DIR)/plan_mlp.md
	@grep -q '/Users/mascit/projects/dnn.c/' $(DOCS_DIR)/plan_mlp.md
	@printf '[docs-check] ok\n'

list:
	@printf 'SRC:\n%s\n' '$(SRC)'
	@printf 'TEST_SRC:\n%s\n' '$(TEST_SRC)'
	@printf 'EXAMPLE_SRC:\n%s\n' '$(EXAMPLE_SRC)'
	@printf 'OBJ:\n%s\n' '$(OBJ)'
	@printf 'TEST_BINS:\n%s\n' '$(TEST_BINS)'
	@printf 'EXAMPLE_BINS:\n%s\n' '$(EXAMPLE_BINS)'

clean:
	@$(RM) $(BUILD_DIR)
	@printf '[clean] removed %s\n' '$(BUILD_DIR)'

# ----- Build rules ----------------------------------------------------------
$(LIB): $(OBJ)
	@mkdir -p $(@D)
ifeq ($(strip $(OBJ)),)
	@printf '[lib] no source files found under %s/; creating placeholder archive\n' '$(SRC_DIR)'
	@printf '!<arch>\n' > $@
else
	$(AR) rcs $@ $(OBJ)
endif

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.m
	@mkdir -p $(@D)
	$(CC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.air: %.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c $< -o $@

$(METALLIB): $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SHADERS))
	@mkdir -p $(@D)
	xcrun -sdk macosx metallib $^ -o $@

$(BUILD_DIR)/$(TEST_DIR)/%: $(TEST_DIR)/%.c $(LIB)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/$(EXAMPLE_DIR)/%: $(EXAMPLE_DIR)/%.c $(LIB)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

-include $(DEP)
