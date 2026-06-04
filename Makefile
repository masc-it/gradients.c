# gradients.c v2 Makefile

SHELL := /bin/bash

BUILD_DIR ?= build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
PROBE_DIR := probes
DOCS_DIR := docs
TOOLS_DIR := tools

LIB_NAME := gradients
LIB := $(BUILD_DIR)/lib$(LIB_NAME).a
CONFIG_STAMP := $(BUILD_DIR)/.build-config

CC ?= cc
OBJC ?= clang
METALC ?= xcrun -sdk macosx metal
METALLIB_TOOL ?= xcrun -sdk macosx metallib
AR ?= ar
RM := rm -rf

CPPFLAGS ?= -I$(INCLUDE_DIR)
CFLAGS ?= -std=c11 -O0 -g3 -Wall -Wextra -Wpedantic -Werror \
          -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
          -Wmissing-prototypes -Wno-unused-parameter
LDFLAGS ?=
LDLIBS ?=
PROBE_CFLAGS ?= -std=c11 -Wall -Wextra -Werror -Wpedantic
TOOL_CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic
PERF_BUILD_DIR ?= build-perf
PERF_CFLAGS ?= -std=c11 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror \
               -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
               -Wmissing-prototypes -Wno-unused-parameter
PERF_OBJCFLAGS ?= -I$(INCLUDE_DIR) -DGD_ENABLE_METAL=1 -O3 -DNDEBUG \
                  -Wall -Wextra -Werror -fobjc-arc
PERF_PROBE_CFLAGS ?= -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -Wpedantic

SAN_FLAGS :=
ifeq ($(SAN),1)
SAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer
CFLAGS += $(SAN_FLAGS)
PROBE_CFLAGS += $(SAN_FLAGS)
LDFLAGS += -fsanitize=address,undefined
endif

UNAME_S := $(shell uname -s)

METAL_BUILD_ARTIFACTS :=
METAL_SRC :=
METAL_AIR :=
TEST_ENV :=

ifeq ($(UNAME_S),Darwin)
CPPFLAGS += -DGD_ENABLE_METAL=1
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -path '$(SRC_DIR)/backends/null/*' 2>/dev/null | sort)
MSRC := $(shell find $(SRC_DIR) -type f -name '*.m' 2>/dev/null | sort)
METAL_SRC := $(shell find $(SRC_DIR)/backends/metal -type f -name '*.metal' 2>/dev/null | sort)
METAL_AIR := $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SRC))
METALLIB := $(BUILD_DIR)/gradients.metallib
METAL_BUILD_ARTIFACTS := $(METALLIB)
TEST_ENV := GRADIENTS_METALLIB=$(METALLIB)
OBJCFLAGS ?= $(CPPFLAGS) -O0 -g3 -Wall -Wextra -Werror -fobjc-arc $(SAN_FLAGS)
LDLIBS += -framework Foundation -framework Metal
else
CPPFLAGS += -DGD_ENABLE_METAL=0
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -path '$(SRC_DIR)/backends/metal/*' 2>/dev/null | sort)
MSRC :=
endif

OBJ := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC)) $(patsubst %.m,$(BUILD_DIR)/%.o,$(MSRC))
DEP := $(OBJ:.o=.d)
OPS_REGISTRY_OBJS := $(filter $(BUILD_DIR)/src/ops/%.o,$(OBJ)) $(filter $(BUILD_DIR)/src/autograd/%.o,$(OBJ))

TEST_SRC := $(shell find $(TEST_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRC))

FOUNDATION_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_foundation_probe
METAL_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_metal_arena_probe
GEMM_PERF_PROBE := $(PERF_BUILD_DIR)/$(PROBE_DIR)/v2_matmul_training_perf_probe
GEN_OPS_TOOL := $(BUILD_DIR)/$(TOOLS_DIR)/gen_ops
NEW_OP_TOOL := $(BUILD_DIR)/$(TOOLS_DIR)/gradients-new-op
OPS_REGISTRY_STAMP := $(BUILD_DIR)/.ops-registry

.PHONY: help all build check test tests docs-check probes foundation-probe metal-probe gemm-perf-probe tools ops-registry clean list FORCE

help:
	@printf '%s\n' 'gradients.c v2 commands:'
	@printf '%s\n' '  make build             build static library'
	@printf '%s\n' '  make tools             build op registry/scaffold tools'
	@printf '%s\n' '  make test              build + run C tests'
	@printf '%s\n' '  make check             docs-check + tests + foundation probe'
	@printf '%s\n' '  make probes            run standalone foundation probe'
	@printf '%s\n' '  make metal-probe       build + run Metal probe on macOS'
	@printf '%s\n' '  make gemm-perf-probe   optimized public API F16 GEMM performance probe'
	@printf '%s\n' '  make SAN=1 check       sanitizer build'
	@printf '%s\n' '  make list              show discovered files'
	@printf '%s\n' '  make clean             remove build dir'

all: check

build: ops-registry $(LIB) $(METAL_BUILD_ARTIFACTS)
	@printf '[build] %s\n' '$(LIB)'

tools: $(GEN_OPS_TOOL) $(NEW_OP_TOOL)
	@printf '[tools] %s %s\n' '$(GEN_OPS_TOOL)' '$(NEW_OP_TOOL)'

ops-registry: $(OPS_REGISTRY_STAMP)

$(OPS_REGISTRY_STAMP): $(GEN_OPS_TOOL) FORCE
	@$(GEN_OPS_TOOL) --stamp $@

src/ops/op_kind.h src/ops/op_registry.c: $(OPS_REGISTRY_STAMP) ;

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
		$(TEST_ENV) "$$t"; \
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

gemm-perf-probe:
	$(MAKE) BUILD_DIR='$(PERF_BUILD_DIR)' CFLAGS='$(PERF_CFLAGS)' OBJCFLAGS='$(PERF_OBJCFLAGS)' build
	@mkdir -p '$(PERF_BUILD_DIR)/$(PROBE_DIR)'
	$(CC) -I$(INCLUDE_DIR) $(PERF_PROBE_CFLAGS) \
	  $(PROBE_DIR)/v2_matmul_training_perf_probe.c \
	  '$(PERF_BUILD_DIR)/lib$(LIB_NAME).a' $(LDFLAGS) \
	  -framework Foundation -framework Metal \
	  -o '$(GEMM_PERF_PROBE)'
	@printf '[probe] %s\n' '$(GEMM_PERF_PROBE)'
	@GRADIENTS_METALLIB='$(PERF_BUILD_DIR)/gradients.metallib' '$(GEMM_PERF_PROBE)'
else
metal-probe:
	@printf '[metal-probe] skipped: macOS required\n'

gemm-perf-probe:
	@printf '[gemm-perf-probe] skipped: macOS required\n'
endif

list:
	@printf 'SRC:\n%s\n' '$(SRC)'
	@printf 'TEST_SRC:\n%s\n' '$(TEST_SRC)'
	@printf 'MSRC:\n%s\n' '$(MSRC)'
	@printf 'METAL_SRC:\n%s\n' '$(METAL_SRC)'
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

$(LIB): $(OPS_REGISTRY_STAMP) $(CONFIG_STAMP) $(OBJ)
	@mkdir -p $(@D)
	@$(RM) $@
	$(AR) rcs $@ $(OBJ)

$(GEN_OPS_TOOL): $(TOOLS_DIR)/gen_ops.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(TOOL_CFLAGS) $< -o $@

$(NEW_OP_TOOL): $(TOOLS_DIR)/gradients-new-op.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(TOOL_CFLAGS) $< -o $@

$(BUILD_DIR)/%.o: %.c $(CONFIG_STAMP) | ops-registry
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.m $(CONFIG_STAMP) | ops-registry
	@mkdir -p $(@D)
	$(OBJC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(OPS_REGISTRY_OBJS): $(OPS_REGISTRY_STAMP)

ifeq ($(UNAME_S),Darwin)
$(METALLIB): $(METAL_AIR)
	@mkdir -p $(@D)
	$(METALLIB_TOOL) $^ -o $@

$(BUILD_DIR)/%.air: %.metal $(CONFIG_STAMP) $(SRC_DIR)/backends/metal/metal_kernel_types.h
	@mkdir -p $(@D)
	$(METALC) -I$(SRC_DIR)/backends/metal -c $< -o $@
endif

$(BUILD_DIR)/$(TEST_DIR)/%: $(TEST_DIR)/%.c $(LIB) $(METAL_BUILD_ARTIFACTS)
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
