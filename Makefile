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
LDLIBS += -pthread -lm
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
BACKEND ?= auto
NVCC ?= nvcc
CUDA_HOME ?= $(shell nvcc_path="$$(command -v $(NVCC) 2>/dev/null)"; if [ -n "$$nvcc_path" ]; then cd "$$(dirname "$$nvcc_path")/.." && pwd; fi)
CUDA_ARCH ?=
CUDA_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && printf '1' || printf '0')
CUDA_ARCH_FLAGS :=
ifneq ($(strip $(CUDA_ARCH)),)
CUDA_ARCH_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
endif
CUDA_LDFLAGS ?= $(if $(CUDA_HOME),-L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib/x86_64-linux-gnu,)
CUDA_LDLIBS ?= $(CUDA_LDFLAGS) -lcudart -lcublas -lcublasLt -lstdc++
NVCCFLAGS ?= -std=c++17 -O0 -g -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Werror

ifeq ($(BACKEND),auto)
ifeq ($(UNAME_S),Darwin)
SELECTED_BACKEND := metal
else ifeq ($(CUDA_AVAILABLE),1)
SELECTED_BACKEND := cuda
else
SELECTED_BACKEND := null
endif
else
SELECTED_BACKEND := $(BACKEND)
endif

ifeq ($(filter $(SELECTED_BACKEND),metal cuda null),)
$(error unsupported BACKEND '$(BACKEND)' (use auto, metal, cuda, null))
endif
ifeq ($(SELECTED_BACKEND),metal)
ifneq ($(UNAME_S),Darwin)
$(error BACKEND=metal requires Darwin)
endif
endif
ifeq ($(SELECTED_BACKEND),cuda)
ifeq ($(CUDA_AVAILABLE),0)
$(error BACKEND=cuda requires nvcc; set NVCC=/path/to/nvcc if needed)
endif
endif

METAL_BUILD_ARTIFACTS :=
METAL_SRC :=
METAL_HEADERS :=
METAL_AIR :=
TEST_ENV :=
CUSRC :=

ifeq ($(SELECTED_BACKEND),metal)
CPPFLAGS += -DGD_ENABLE_METAL=1 -DGD_ENABLE_CUDA=0
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -name 'perf_test.c' ! -path '$(SRC_DIR)/backends/null/*' ! -path '$(SRC_DIR)/backends/cuda/*' 2>/dev/null | sort)
MSRC := $(shell find $(SRC_DIR) -type f -name '*.m' 2>/dev/null | sort)
METAL_SRC := $(shell find $(SRC_DIR)/backends/metal $(SRC_DIR)/ops $(SRC_DIR)/optim -type f -name '*.metal' 2>/dev/null | sort)
METAL_HEADERS := $(shell find $(SRC_DIR)/backends/metal $(SRC_DIR)/ops $(SRC_DIR)/optim -type f \( -name '*.h' -o -name '*.inc' \) 2>/dev/null | sort)
METAL_AIR := $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SRC))
METALLIB := $(BUILD_DIR)/gradients.metallib
METAL_BUILD_ARTIFACTS := $(METALLIB)
TEST_ENV := GRADIENTS_METALLIB=$(METALLIB)
OBJCFLAGS ?= $(CPPFLAGS) -O0 -g3 -Wall -Wextra -Werror -fobjc-arc $(SAN_FLAGS)
LDLIBS += -framework Foundation -framework Metal
else ifeq ($(SELECTED_BACKEND),cuda)
CPPFLAGS += -DGD_ENABLE_METAL=0 -DGD_ENABLE_CUDA=1 -D_POSIX_C_SOURCE=200809L
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -name 'perf_test.c' ! -path '$(SRC_DIR)/backends/metal/*' ! -path '$(SRC_DIR)/backends/null/*' ! -path '$(SRC_DIR)/backends/cuda/*' 2>/dev/null | sort)
SRC += $(shell find $(SRC_DIR)/backends/cuda -type f -name '*.c' 2>/dev/null | sort)
MSRC :=
CUSRC := $(shell find $(SRC_DIR)/backends/cuda $(SRC_DIR)/ops $(SRC_DIR)/optim -type f -name '*.cu' 2>/dev/null | sort)
LDLIBS += $(CUDA_LDLIBS)
else
CPPFLAGS += -DGD_ENABLE_METAL=0 -DGD_ENABLE_CUDA=0 -D_POSIX_C_SOURCE=200809L
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' ! -name 'perf_test.c' ! -path '$(SRC_DIR)/backends/metal/*' ! -path '$(SRC_DIR)/backends/cuda/*' 2>/dev/null | sort)
MSRC :=
endif

OBJ := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC)) $(patsubst %.m,$(BUILD_DIR)/%.o,$(MSRC)) $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUSRC))
DEP := $(OBJ:.o=.d)
OPS_REGISTRY_OBJS := $(filter $(BUILD_DIR)/src/ops/%.o,$(OBJ)) $(filter $(BUILD_DIR)/src/autograd/%.o,$(OBJ))

TEST_SRC := $(shell find $(TEST_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRC))

FOUNDATION_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_foundation_probe
METAL_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_metal_arena_probe
CUDA_PROBE := $(BUILD_DIR)/$(PROBE_DIR)/v2_cuda_backend_smoke
GEMM_PERF_PROBE := $(PERF_BUILD_DIR)/$(PROBE_DIR)/v2_matmul_training_perf_probe
ELEM_PERF_PROBE := $(PERF_BUILD_DIR)/$(PROBE_DIR)/v2_elementwise_reduce_perf_probe
GEN_OPS_TOOL := $(BUILD_DIR)/$(TOOLS_DIR)/gen_ops
NEW_OP_TOOL := $(BUILD_DIR)/$(TOOLS_DIR)/gradients-new-op
TOKENIZE_TOOL := $(BUILD_DIR)/$(TOOLS_DIR)/gradients-tokenize
TOKENIZER_SRC := $(shell find $(SRC_DIR)/tokenizer -type f -name '*.c' 2>/dev/null | sort) $(SRC_DIR)/core/status.c
OPS_REGISTRY_STAMP := $(BUILD_DIR)/.ops-registry
OP_UPPER = $(shell printf '%s' '$(OP)' | tr '[:lower:]' '[:upper:]')

.PHONY: help all build check test tests docs-check probes foundation-probe metal-probe cuda-probe gemm-perf-probe elementwise-reduce-perf-probe op-perf tools ops-registry clean list FORCE

help:
	@printf '%s\n' 'gradients.c v2 commands:'
	@printf '%s\n' '  make build             build static library'
	@printf '%s\n' '  make BACKEND=cuda build  build with CUDA backend'
	@printf '%s\n' '  make tools             build op registry/scaffold/tokenizer tools'
	@printf '%s\n' '  make test              build + run C tests'
	@printf '%s\n' '  make check             docs-check + tests + foundation probe'
	@printf '%s\n' '  make probes            run standalone foundation probe'
	@printf '%s\n' '  make metal-probe       build + run Metal probe on macOS'
	@printf '%s\n' '  make BACKEND=cuda cuda-probe build + run CUDA backend smoke probe'
	@printf '%s\n' '  make gemm-perf-probe   optimized public API F16 GEMM performance probe'
	@printf '%s\n' '  make elementwise-reduce-perf-probe optimized binary/reduce performance probe'
	@printf '%s\n' '  make op-perf OP=relu   optimized src/ops/<op>/perf_test.c probe'
	@printf '%s\n' '  make SAN=1 check       sanitizer build'
	@printf '%s\n' '  make list              show discovered files'
	@printf '%s\n' '  make clean             remove build dir'

all: check

build: ops-registry $(LIB) $(METAL_BUILD_ARTIFACTS)
	@printf '[build] %s\n' '$(LIB)'

tools: $(GEN_OPS_TOOL) $(NEW_OP_TOOL) $(TOKENIZE_TOOL)
	@printf '[tools] %s %s %s\n' '$(GEN_OPS_TOOL)' '$(NEW_OP_TOOL)' '$(TOKENIZE_TOOL)'

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

ifeq ($(SELECTED_BACKEND),cuda)
cuda-probe: build $(CUDA_PROBE)
	@printf '[probe] %s\n' '$(CUDA_PROBE)'
	@$(CUDA_PROBE)
else
cuda-probe:
	@printf '[cuda-probe] skipped: use BACKEND=cuda\n'
endif

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

elementwise-reduce-perf-probe:
	$(MAKE) BUILD_DIR='$(PERF_BUILD_DIR)' CFLAGS='$(PERF_CFLAGS)' OBJCFLAGS='$(PERF_OBJCFLAGS)' build
	@mkdir -p '$(PERF_BUILD_DIR)/$(PROBE_DIR)'
	$(CC) -I$(INCLUDE_DIR) $(PERF_PROBE_CFLAGS) \
	  $(PROBE_DIR)/v2_elementwise_reduce_perf_probe.c \
	  '$(PERF_BUILD_DIR)/lib$(LIB_NAME).a' $(LDFLAGS) \
	  -framework Foundation -framework Metal \
	  -o '$(ELEM_PERF_PROBE)'
	@printf '[probe] %s\n' '$(ELEM_PERF_PROBE)'
	@GRADIENTS_METALLIB='$(PERF_BUILD_DIR)/gradients.metallib' '$(ELEM_PERF_PROBE)'

op-perf:
	@if [ -z '$(OP)' ]; then \
	  printf '%s\n' 'usage: make op-perf OP=<op-name>'; \
	  exit 2; \
	fi
	@if [ ! -f '$(SRC_DIR)/ops/$(OP)/perf_test.c' ]; then \
	  printf 'missing %s\n' '$(SRC_DIR)/ops/$(OP)/perf_test.c'; \
	  exit 2; \
	fi
	$(MAKE) BUILD_DIR='$(PERF_BUILD_DIR)' CFLAGS='$(PERF_CFLAGS)' OBJCFLAGS='$(PERF_OBJCFLAGS)' build
	@mkdir -p '$(PERF_BUILD_DIR)/$(SRC_DIR)/ops/$(OP)'
	$(CC) -I$(INCLUDE_DIR) $(PERF_PROBE_CFLAGS) -DGD_PERF_TEST_MAIN -DGD_$(OP_UPPER)_PERF_TEST_MAIN \
	  '$(SRC_DIR)/ops/$(OP)/perf_test.c' \
	  '$(PERF_BUILD_DIR)/lib$(LIB_NAME).a' $(LDFLAGS) \
	  -framework Foundation -framework Metal \
	  -o '$(PERF_BUILD_DIR)/$(SRC_DIR)/ops/$(OP)/perf_test'
	@printf '[probe] %s\n' '$(PERF_BUILD_DIR)/$(SRC_DIR)/ops/$(OP)/perf_test'
	@GRADIENTS_METALLIB='$(PERF_BUILD_DIR)/gradients.metallib' '$(PERF_BUILD_DIR)/$(SRC_DIR)/ops/$(OP)/perf_test'
else
metal-probe:
	@printf '[metal-probe] skipped: macOS required\n'

gemm-perf-probe:
	@printf '[gemm-perf-probe] skipped: macOS required\n'

elementwise-reduce-perf-probe:
	@printf '[elementwise-reduce-perf-probe] skipped: macOS required\n'

op-perf:
	@printf '[op-perf] skipped: macOS required\n'
endif

list:
	@printf 'BACKEND: %s (selected: %s)\n' '$(BACKEND)' '$(SELECTED_BACKEND)'
	@printf 'SRC:\n%s\n' '$(SRC)'
	@printf 'TEST_SRC:\n%s\n' '$(TEST_SRC)'
	@printf 'MSRC:\n%s\n' '$(MSRC)'
	@printf 'CUSRC:\n%s\n' '$(CUSRC)'
	@printf 'METAL_SRC:\n%s\n' '$(METAL_SRC)'
	@printf 'METAL_HEADERS:\n%s\n' '$(METAL_HEADERS)'
	@printf 'OBJ:\n%s\n' '$(OBJ)'
	@printf 'TEST_BINS:\n%s\n' '$(TEST_BINS)'

clean:
	@$(RM) $(BUILD_DIR)
	@printf '[clean] removed %s\n' '$(BUILD_DIR)'

FORCE:

$(CONFIG_STAMP): FORCE
	@mkdir -p $(@D)
	@tmp="$@.tmp"; \
	printf 'SAN=%s\nUNAME_S=%s\nBACKEND=%s\nSELECTED_BACKEND=%s\nCPPFLAGS=%s\nCFLAGS=%s\nOBJCFLAGS=%s\nNVCC=%s\nNVCCFLAGS=%s\nCUDA_ARCH=%s\n' '$(SAN)' '$(UNAME_S)' '$(BACKEND)' '$(SELECTED_BACKEND)' '$(CPPFLAGS)' '$(CFLAGS)' '$(OBJCFLAGS)' '$(NVCC)' '$(NVCCFLAGS)' '$(CUDA_ARCH)' > "$$tmp"; \
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

$(TOKENIZE_TOOL): $(TOOLS_DIR)/gradients-tokenize.c $(TOKENIZER_SRC) include/gradients/tokenizer.h include/gradients/status.h $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(TOOL_CFLAGS) $< $(TOKENIZER_SRC) -o $@

$(BUILD_DIR)/%.o: %.c $(CONFIG_STAMP) | ops-registry
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.m $(CONFIG_STAMP) | ops-registry
	@mkdir -p $(@D)
	$(OBJC) $(OBJCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.cu $(CONFIG_STAMP) | ops-registry
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) -MMD -MP -c $< -o $@

$(OPS_REGISTRY_OBJS): $(OPS_REGISTRY_STAMP)

ifeq ($(UNAME_S),Darwin)
$(METALLIB): $(METAL_AIR)
	@mkdir -p $(@D)
	$(METALLIB_TOOL) $^ -o $@

$(BUILD_DIR)/%.air: %.metal $(CONFIG_STAMP) $(METAL_HEADERS)
	@mkdir -p $(@D)
	$(METALC) -I$(SRC_DIR) -I$(SRC_DIR)/backends/metal -c $< -o $@
endif

$(BUILD_DIR)/$(TEST_DIR)/%: $(TEST_DIR)/%.c $(LIB) $(METAL_BUILD_ARTIFACTS)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(FOUNDATION_PROBE): $(PROBE_DIR)/v2_foundation_probe.c $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(CC) $(PROBE_CFLAGS) $< $(LDFLAGS) -o $@

$(CUDA_PROBE): $(PROBE_DIR)/v2_cuda_backend_smoke.c $(LIB)
	@mkdir -p $(@D)
	$(CC) -I. $(CPPFLAGS) $(PROBE_CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(METAL_PROBE): $(PROBE_DIR)/v2_metal_arena_probe.m $(CONFIG_STAMP)
	@mkdir -p $(@D)
	$(OBJC) -fobjc-arc -Wall -Wextra -Werror \
	  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
	  $< -o $@

-include $(DEP)
