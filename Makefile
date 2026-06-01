# gradients.c Makefile
# Primary commands build first, then run their workload.
# Examples:
#   make test        # build lib + tests, then run tests
#   make mlp         # build lib + examples/mlp/mlp, then run MLP example
#   make check       # build, run docs/size checks, run tests

SHELL := /bin/bash

# ----- Project layout -------------------------------------------------------
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests
EXAMPLE_DIR := examples
TOOL_DIR := tools
DOCS_DIR := docs

LIB_NAME := gradients
LIB := $(BUILD_DIR)/lib$(LIB_NAME).a

# ----- Toolchain ------------------------------------------------------------
CC ?= cc
AR ?= ar
RM := rm -rf

CPPFLAGS ?= -I$(INCLUDE_DIR)
CPPFLAGS += -I$(BUILD_DIR)/generated
CFLAGS ?= -std=c11 -O0 -g3 -Wall -Wextra -Wpedantic -Werror \
          -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
          -Wmissing-prototypes -Wno-unused-parameter
BENCH_CFLAGS ?= -std=c11 -O3 -ffast-math -DNDEBUG -Wall -Wextra -Wpedantic -Werror \
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
# Let C/Obj-C sources conditionally compile the Metal registration path.
CPPFLAGS += -DGD_ENABLE_METAL=1
OBJCFLAGS := $(CPPFLAGS) $(CFLAGS) -fobjc-arc -x objective-c
LDLIBS += -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework QuartzCore
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

# ----- Generated operator registry inputs -----------------------------------
OP_CORE_FILES := $(sort $(wildcard $(SRC_DIR)/ops/*/core_*.c))
OP_GRAD_FILES := $(sort $(wildcard $(SRC_DIR)/ops/*/grad_*.c))
CPU_OP_FILES := $(sort $(wildcard $(SRC_DIR)/ops/*/cpu_*.c))
METAL_OP_FILES := $(sort $(wildcard $(SRC_DIR)/ops/*/metal_*.m))
METAL_OP_SHADER_FILES := $(sort $(wildcard $(SRC_DIR)/ops/*/metal_*.metal))

GENERATED_DIR := $(BUILD_DIR)/generated
GEN_OPS_BIN := $(BUILD_DIR)/tools/gen_ops
GEN_OPS_STAMP := $(GENERATED_DIR)/.gen_ops.stamp
GENERATED_FILES := \
  $(GENERATED_DIR)/op_kind.h \
  $(GENERATED_DIR)/op_registry.inc \
  $(GENERATED_DIR)/bwd_registry.inc \
  $(GENERATED_DIR)/cpu_registry.inc \
  $(GENERATED_DIR)/metal_registry.inc \
  $(GENERATED_DIR)/metal_shaders.mk \
  $(GENERATED_DIR)/op_matrix.md

# gpt_bench is a profiling harness, not a correctness test: exclude it from the
# auto-discovered test runner (build/run via `make gpt-bench`).
GPT_BENCH_SRC := $(TEST_DIR)/gpt_bench.c
GPT_BENCH_BIN := $(BUILD_DIR)/$(TEST_DIR)/gpt_bench
TEST_SRC := $(filter-out $(GPT_BENCH_SRC),\
             $(shell find $(TEST_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort))
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/$(TEST_DIR)/%,$(TEST_SRC))

EXAMPLE_SRC := $(shell find $(EXAMPLE_DIR) -type f -name '*.c' 2>/dev/null | sort)
EXAMPLE_BINS := $(patsubst %.c,$(BUILD_DIR)/%,$(EXAMPLE_SRC))
TOOL_SRC := $(shell find $(TOOL_DIR) -maxdepth 1 -type f -name '*.c' 2>/dev/null | sort)
TOOL_BINS := $(patsubst %.c,$(BUILD_DIR)/%,$(TOOL_SRC))

MLP_SRC := $(EXAMPLE_DIR)/mlp/mlp.c
MLP_BIN := $(BUILD_DIR)/$(EXAMPLE_DIR)/mlp/mlp
GPT_SRC := $(EXAMPLE_DIR)/gpt/gpt.c
GPT_BIN := $(BUILD_DIR)/$(EXAMPLE_DIR)/gpt/gpt

# ----- Public commands ------------------------------------------------------
.PHONY: help all build check test tests mlp gpt bench-gpt gpt-bench _gpt_bench_run examples tools docs-check size-check-report size-check generated clean list

help:
	@printf '%s\n' 'gradients.c commands:'
	@printf '%s\n' '  make build       build library'
	@printf '%s\n' '  make test        build library + tests, then run all tests'
	@printf '%s\n' '  make check       build, run docs/size checks, then run tests'
	@printf '%s\n' '  make size-check  warn >800 LOC and fail new non-allowlisted >1000 LOC'
	@printf '%s\n' '  make generated   regenerate operator registry scaffolding'
	@printf '%s\n' '  make mlp         build library + MLP example, then run it'
	@printf '%s\n' '  make gpt         build library + GPT example, then run it (GD_DEVICE=metal for GPU)'
	@printf '%s\n' '  make bench-gpt   release build under build-release/, then run GPT example'
	@printf '%s\n' '  make gpt-bench   release build, then run GPT profiling harness (GD_DEVICE, GD_BENCH_*)'
	@printf '%s\n' '  make examples    build library + all examples, then run all examples'
	@printf '%s\n' '  make tools       build command-line tools'
	@printf '%s\n' '  make docs-check  build, then validate docs links/references'
	@printf '%s\n' '  make list        show discovered source/test/example files'
	@printf '%s\n' '  make clean       remove build artifacts'
	@printf '%s\n' ''
	@printf '%s\n' 'Primary run commands intentionally build first.'

all: check

build: $(LIB) $(METALLIB)
	@printf '[build] ok (metal=%s)\n' '$(GD_ENABLE_METAL)'

check: build docs-check size-check test tools
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

gpt: build
ifeq ($(wildcard $(GPT_SRC)),)
	@printf '[gpt] missing %s; skipped\n' '$(GPT_SRC)'
else
	@$(MAKE) --no-print-directory $(GPT_BIN)
	@printf '[gpt] run %s (GD_DEVICE=%s)\n' '$(GPT_BIN)' '$(GD_DEVICE)'
	@$(GPT_BIN)
endif

bench-gpt:
	@$(MAKE) --no-print-directory BUILD_DIR=build-release CFLAGS='$(BENCH_CFLAGS)' gpt

# Release-built GPT profiling harness. Honors GD_DEVICE and GD_BENCH_* env knobs.
gpt-bench:
	@$(MAKE) --no-print-directory BUILD_DIR=build-release CFLAGS='$(BENCH_CFLAGS)' _gpt_bench_run

_gpt_bench_run: build $(GPT_BENCH_BIN)
	@printf '[gpt-bench] run %s (GD_DEVICE=%s)\n' '$(GPT_BENCH_BIN)' '$(GD_DEVICE)'
	@GRADIENTS_METALLIB=$(BUILD_DIR)/gradients.metallib $(GPT_BENCH_BIN)

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

tools: build $(TOOL_BINS)
ifeq ($(strip $(TOOL_BINS)),)
	@printf '[tools] no tools found under %s/ yet\n' '$(TOOL_DIR)'
else
	@printf '[tools] built %s\n' '$(TOOL_BINS)'
endif

docs-check: build
	@test -f $(DOCS_DIR)/design_spec.md
	@test -f $(DOCS_DIR)/plan_mlp.md
	@grep -q 'design_spec.md' $(DOCS_DIR)/plan_mlp.md
	@grep -q '/Users/mascit/projects/dnn.c/' $(DOCS_DIR)/plan_mlp.md
	@printf '[docs-check] ok\n'

size-check-report:
	@set -euo pipefail; \
	found=0; \
	while IFS= read -r f; do \
		n=$$(wc -l < "$$f" | tr -d ' '); \
		if (( n > 1000 )); then \
			printf 'size-check-report: FAIL-threshold %s has %s lines\n' "$$f" "$$n"; \
			found=1; \
		elif (( n > 800 )); then \
			printf 'size-check-report: WARN-threshold %s has %s lines\n' "$$f" "$$n"; \
			found=1; \
		fi; \
	done < <(find $(SRC_DIR) $(INCLUDE_DIR) -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' -o -name '*.cu' \) | sort); \
	if (( found == 0 )); then printf '[size-check-report] no files over 800 LOC\n'; fi

size-check:
	@set -euo pipefail; \
	allowlist='$(DOCS_DIR)/size_allowlist.txt'; \
	test -f "$$allowlist"; \
	fail=0; \
	while IFS= read -r f; do \
		n=$$(wc -l < "$$f" | tr -d ' '); \
		if (( n > 1000 )); then \
			if awk -v p="$$f" '($$0 !~ /^[[:space:]]*(#|$$)/ && $$1 == p) { found = 1 } END { exit found ? 0 : 1 }' "$$allowlist"; then \
				printf 'size-check: WARN allowlisted %s has %s lines\n' "$$f" "$$n"; \
			else \
				printf 'size-check: FAIL %s has %s lines (>1000) and is not allowlisted\n' "$$f" "$$n"; \
				fail=1; \
			fi; \
		elif (( n > 800 )); then \
			printf 'size-check: WARN %s has %s lines\n' "$$f" "$$n"; \
		fi; \
	done < <(find $(SRC_DIR) $(INCLUDE_DIR) -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' -o -name '*.cu' \) | sort); \
	exit $$fail

list:
	@printf 'SRC:\n%s\n' '$(SRC)'
	@printf 'TEST_SRC:\n%s\n' '$(TEST_SRC)'
	@printf 'EXAMPLE_SRC:\n%s\n' '$(EXAMPLE_SRC)'
	@printf 'TOOL_SRC:\n%s\n' '$(TOOL_SRC)'
	@printf 'OBJ:\n%s\n' '$(OBJ)'
	@printf 'TEST_BINS:\n%s\n' '$(TEST_BINS)'
	@printf 'EXAMPLE_BINS:\n%s\n' '$(EXAMPLE_BINS)'
	@printf 'TOOL_BINS:\n%s\n' '$(TOOL_BINS)'

clean:
	@$(RM) $(BUILD_DIR)
	@printf '[clean] removed %s\n' '$(BUILD_DIR)'

# ----- Build rules ----------------------------------------------------------
generated: $(GEN_OPS_STAMP)
	@printf '[generated] ok\n'

$(GEN_OPS_BIN): tools/gen_ops.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $<

$(GEN_OPS_STAMP): $(GEN_OPS_BIN) $(OP_CORE_FILES) $(OP_GRAD_FILES) $(CPU_OP_FILES) $(METAL_OP_FILES) $(METAL_OP_SHADER_FILES)
	@mkdir -p $(GENERATED_DIR)
	@$< \
	  --out $(GENERATED_DIR) \
	  --core $(OP_CORE_FILES) \
	  --grad $(OP_GRAD_FILES) \
	  --cpu $(CPU_OP_FILES) \
	  --metal $(METAL_OP_FILES) \
	  --metal-shaders $(METAL_OP_SHADER_FILES)
	@touch $@

$(GENERATED_FILES): $(GEN_OPS_STAMP)
	@test -f $@

$(LIB): $(OBJ)
	@mkdir -p $(@D)
ifeq ($(strip $(OBJ)),)
	@printf '[lib] no source files found under %s/; creating placeholder archive\n' '$(SRC_DIR)'
	@printf '!<arch>\n' > $@
else
	$(AR) rcs $@ $(OBJ)
endif

$(BUILD_DIR)/%.o: %.c $(GEN_OPS_STAMP)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.m $(GEN_OPS_STAMP)
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

$(BUILD_DIR)/$(TOOL_DIR)/%: $(TOOL_DIR)/%.c $(LIB)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

-include $(DEP)
