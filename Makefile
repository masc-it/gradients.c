# gradients.c v2 Makefile

SHELL := /bin/bash

BUILD_DIR ?= build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
PROBE_DIR := probes
DOCS_DIR := docs

LIB_NAME := gradients
LIB := $(BUILD_DIR)/lib$(LIB_NAME).a
CONFIG_STAMP := $(BUILD_DIR)/.build-config

CC ?= cc
OBJC ?= clang
AR ?= ar
RM := rm -rf

CPPFLAGS ?= -I$(INCLUDE_DIR)
CFLAGS ?= -std=c11 -O0 -g3 -Wall -Wextra -Wpedantic -Werror \
          -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
          -Wmissing-prototypes -Wno-unused-parameter
LDFLAGS ?=
LDLIBS ?=
PROBE_CFLAGS ?= -std=c11 -Wall -Wextra -Werror -Wpedantic

SAN_FLAGS :=
ifeq ($(SAN),1)
SAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(SAN_FLAGS)
PROBE_CFLAGS += $(SAN_FLAGS)
LDFLAGS += -fsanitize=address,undefined
endif

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
CPPFLAGS += -DGD_ENABLE_METAL=1
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -path '$(SRC_DIR)/backends/null/*' 2>/dev/null | sort)
MSRC := $(shell find $(SRC_DIR) -type f -name '*.m' 2>/dev/null | sort)
OBJCFLAGS ?= $(CPPFLAGS) -O0 -g3 -Wall -Wextra -Werror -fobjc-arc $(SAN_FLAGS)
LDLIBS += -framework Foundation -framework Metal
else
CPPFLAGS += -DGD_ENABLE_METAL=0
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -path '$(SRC_DIR)/backends/metal/*' 2>/dev/null | sort)
MSRC :=
endif

OBJ := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC)) $(patsubst %.m,$(BUILD_DIR)/%.o,$(MSRC))
DEP := $(OBJ:.o=.d)

TEST_SRC := $(shell find $(TEST_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRC))

FOUNDATION_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_foundation_probe
METAL_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_metal_arena_probe

.PHONY: help all build check test tests docs-check probes foundation-probe metal-probe clean list FORCE

help:
	@printf '%s\n' 'gradients.c v2 commands:'
	@printf '%s\n' '  make build             build static library'
	@printf '%s\n' '  make test              build + run C tests'
	@printf '%s\n' '  make check             docs-check + tests + foundation probe'
	@printf '%s\n' '  make probes            run standalone foundation probe'
	@printf '%s\n' '  make metal-probe       build + run Metal probe on macOS'
	@printf '%s\n' '  make SAN=1 check       sanitizer build'
	@printf '%s\n' '  make list              show discovered files'
	@printf '%s\n' '  make clean             remove build dir'

all: check

build: $(LIB)
	@printf '[build] %s\n' '$(LIB)'

check: docs-check test foundation-probe
	@printf '[check] ok\n'

docs-check:
	@test -f $(DOCS_DIR)/design_spec.md
	@grep -q '^## Memory model' $(DOCS_DIR)/design_spec.md
	@grep -q 'sealed' $(DOCS_DIR)/design_spec.md
	@printf '[docs-check] ok\n'

test tests: build $(TEST_BINS)
ifeq ($(strip $(TEST_BINS)),)
	@printf '[test] no tests found\n'
else
	@set -euo pipefail; \
	for t in $(TEST_BINS); do \
		printf '[test] %s\n' "$$t"; \
		"$$t"; \
	done
endif
	@printf '[test] ok\n'

probes: foundation-probe

foundation-probe: $(FOUNDATION_PROBE)
	@printf '[probe] %s\n' '$(FOUNDATION_PROBE)'
	@$(FOUNDATION_PROBE)

ifeq ($(UNAME_S),Darwin)
metal-probe: $(METAL_PROBE)
	@printf '[probe] %s\n' '$(METAL_PROBE)'
	@$(METAL_PROBE)
else
metal-probe:
	@printf '[metal-probe] skipped: macOS required\n'
endif

list:
	@printf 'SRC:\n%s\n' '$(SRC)'
	@printf 'TEST_SRC:\n%s\n' '$(TEST_SRC)'
	@printf 'MSRC:\n%s\n' '$(MSRC)'
	@printf 'OBJ:\n%s\n' '$(OBJ)'
	@printf 'TEST_BINS:\n%s\n' '$(TEST_BINS)'

clean:
	@$(RM) $(BUILD_DIR)
	@printf '[clean] removed %s\n' '$(BUILD_DIR)'

FORCE:

$(CONFIG_STAMP): FORCE
	@mkdir -p $(@D)
	@tmp="$@.tmp"; \
	printf 'SAN=%s\nUNAME_S=%s\nCFLAGS=%s\nOBJCFLAGS=%s\n' '$(SAN)' '$(UNAME_S)' '$(CFLAGS)' '$(OBJCFLAGS)' > "$$tmp"; \
	if ! test -f "$@" || ! cmp -s "$$tmp" "$@"; then \
		mv "$$tmp" "$@"; \
	else \
		rm "$$tmp"; \
	fi

$(LIB): $(CONFIG_STAMP) $(OBJ)
	@mkdir -p $(@D)
	@$(RM) $@
	$(AR) rcs $@ $(OBJ)

$(BUILD_DIR)/%.o: %.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.m $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(OBJC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/$(TEST_DIR)/%: $(TEST_DIR)/%.c $(LIB)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(FOUNDATION_PROBE): $(PROBE_DIR)/v2_foundation_probe.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(PROBE_CFLAGS) $< $(LDFLAGS) -o $@

$(METAL_PROBE): $(PROBE_DIR)/v2_metal_arena_probe.m $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(OBJC) -fobjc-arc -Wall -Wextra -Werror \
	  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
	  $< -o $@

-include $(DEP)
