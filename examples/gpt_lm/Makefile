ROOT := ../..
TRAIN_BIN := gd_main_gpt_lm
INFER_BIN := gd_main_infer_gpt_lm
GENERATE_BIN := gd_main_generate_gpt_lm
BIN := $(TRAIN_BIN)
CONFIG ?= config.yaml
SHARED_SRC := gpt_lm_shared.c
DATA_DIR ?= data
DATA_STAMP := $(DATA_DIR)/.gpt_lm_gdds.stamp
DATA_ARGS_FILE := $(DATA_DIR)/.gpt_lm_gdds.args
BUILD_ARGS_FILE := .gpt_lm_build.args
GPT_LM_DATASET ?= ita_dict

ifeq ($(GPT_LM_DATASET),ita_dict)
DATA_SCRIPT := dataset_ita_dict.py
DATA_FORMAT_VERSION := gpt-lm-ita-dict-v3
else ifeq ($(GPT_LM_DATASET),ita_dict_v2)
DATA_SCRIPT := dataset_ita_dict_v2.py
DATA_FORMAT_VERSION := gpt-lm-ita-dict-v2-md-gemma-docs-v6-compact-u16-segments
else ifeq ($(GPT_LM_DATASET),promessi)
DATA_SCRIPT := dataset.py
DATA_FORMAT_VERSION := gpt-lm-promessi-v3
else
$(error unsupported GPT_LM_DATASET=$(GPT_LM_DATASET); expected promessi, ita_dict, or ita_dict_v2)
endif

CC ?= cc
PYTHON ?= uv run --with numpy
UNAME_S := $(shell uname -s)
CPPFLAGS := -I$(ROOT)/include
ifneq ($(GPT_LM_VOCAB_SIZE),)
CPPFLAGS += -DGPT_VOCAB_SIZE=$(GPT_LM_VOCAB_SIZE)
endif
ifneq ($(GPT_LM_CONTEXT_LENGTH),)
CPPFLAGS += -DGPT_CONTEXT_LENGTH=$(GPT_LM_CONTEXT_LENGTH)
endif
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic
CORE_CFLAGS ?= -std=c11 -O2 -g3 -Wall -Wextra -Wpedantic -Werror \
               -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes \
               -Wmissing-prototypes -Wno-unused-parameter
CORE_OBJCFLAGS ?= $$(CPPFLAGS) -O2 -g3 -Wall -Wextra -Werror -fobjc-arc
include ../common.mk
LDLIBS := -pthread -lm

ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework Foundation -framework Metal
endif

BUILD_ARGS := GPT_LM_VOCAB_SIZE=$(GPT_LM_VOCAB_SIZE) GPT_LM_CONTEXT_LENGTH=$(GPT_LM_CONTEXT_LENGTH)

DATA_ARGS := --out-dir $(DATA_DIR) --preview-samples 0
ifneq ($(GPT_LM_VOCAB_SIZE),)
DATA_ARGS += --vocab-size $(GPT_LM_VOCAB_SIZE)
endif
ifneq ($(GPT_LM_CONTEXT_LENGTH),)
DATA_ARGS += --context-length $(GPT_LM_CONTEXT_LENGTH)
endif
ifneq ($(GPT_LM_MAX_SHARD_BYTES),)
DATA_ARGS += --max-shard-bytes $(GPT_LM_MAX_SHARD_BYTES)
endif
ifneq ($(GPT_LM_VAL_FRACTION),)
DATA_ARGS += --val-fraction $(GPT_LM_VAL_FRACTION)
endif
ifneq ($(GPT_LM_SPLIT_SEED),)
DATA_ARGS += --split-seed $(GPT_LM_SPLIT_SEED)
endif

ifeq ($(GPT_LM_DATASET),promessi)
ifneq ($(GPT_LM_SOURCE),)
DATA_ARGS += --source $(GPT_LM_SOURCE)
endif
ifneq ($(GPT_LM_VAL_SAMPLES),)
DATA_ARGS += --val-samples $(GPT_LM_VAL_SAMPLES)
endif
ifneq ($(GPT_LM_SPLIT_BLOCK_CHARS),)
DATA_ARGS += --split-block-chars $(GPT_LM_SPLIT_BLOCK_CHARS)
endif
ifneq ($(GPT_LM_CHARS_PER_TOKEN_ESTIMATE),)
DATA_ARGS += --chars-per-token-estimate $(GPT_LM_CHARS_PER_TOKEN_ESTIMATE)
endif
ifneq ($(GPT_LM_RETRAIN_TOKENIZER),)
DATA_ARGS += --retrain-tokenizer
endif
else ifeq ($(GPT_LM_DATASET),ita_dict)
ifneq ($(GPT_LM_ITA_CLEAN_DIR),)
DATA_ARGS += --clean-dir $(GPT_LM_ITA_CLEAN_DIR)
endif
ifneq ($(GPT_LM_ITA_ENRICHED_DIR),)
DATA_ARGS += --enriched-dir $(GPT_LM_ITA_ENRICHED_DIR)
endif
ifneq ($(GPT_LM_MIN_FREQUENCY),)
DATA_ARGS += --min-frequency $(GPT_LM_MIN_FREQUENCY)
endif
ifneq ($(GPT_LM_MAX_EXAMPLES_PER_DEFINITION),)
DATA_ARGS += --max-examples-per-definition $(GPT_LM_MAX_EXAMPLES_PER_DEFINITION)
endif
else ifeq ($(GPT_LM_DATASET),ita_dict_v2)
ifneq ($(GPT_LM_ITA_SOURCE_DIR),)
DATA_ARGS += --source-dir $(GPT_LM_ITA_SOURCE_DIR)
endif
ifneq ($(GPT_LM_ITA_SOURCE_GLOB),)
DATA_ARGS += --source-glob $(GPT_LM_ITA_SOURCE_GLOB)
endif
ifneq ($(GPT_LM_ITA_STORY_SUFFIX),)
DATA_ARGS += --story-suffix $(GPT_LM_ITA_STORY_SUFFIX)
endif
ifneq ($(GPT_LM_ITA_QUIZ_SUFFIX),)
DATA_ARGS += --quiz-suffix $(GPT_LM_ITA_QUIZ_SUFFIX)
endif
ifneq ($(GPT_LM_MIN_FREQUENCY),)
DATA_ARGS += --min-frequency $(GPT_LM_MIN_FREQUENCY)
endif
ifneq ($(GPT_LM_MAX_EXAMPLES_PER_DEFINITION),)
DATA_ARGS += --max-examples-per-definition $(GPT_LM_MAX_EXAMPLES_PER_DEFINITION)
endif
ifneq ($(GPT_LM_KEEP_NONE_FIELDS),)
DATA_ARGS += --keep-none-fields
endif
endif

.PHONY: all run infer interactive build-lib tools data data-tools clean smoke overfit-smoke generate-smoke periodic-generate-smoke checkpoint-smoke FORCE

all: $(TRAIN_BIN) $(INFER_BIN) $(GENERATE_BIN)

data: $(DATA_STAMP)

data-tools:
	$(MAKE) -C $(ROOT) tools

$(BUILD_ARGS_FILE): FORCE
	@args='$(BUILD_ARGS)'; \
	if [ ! -f '$(BUILD_ARGS_FILE)' ] || [ "$$(cat '$(BUILD_ARGS_FILE)')" != "$$args" ]; then \
		printf '%s\n' "$$args" > '$(BUILD_ARGS_FILE)'; \
	fi

$(DATA_STAMP): FORCE $(DATA_SCRIPT) $(ROOT)/tools/gdds_utils.py $(ROOT)/tools/tok_utils.py data-tools
	@mkdir -p $(DATA_DIR)
	@args='$(DATA_FORMAT_VERSION) $(DATA_ARGS)'; \
	if [ -f '$(DATA_STAMP)' ] && [ -f '$(DATA_ARGS_FILE)' ] && \
	   [ "$$(cat '$(DATA_ARGS_FILE)')" = "$$args" ] && \
	   [ '$(DATA_STAMP)' -nt '$(DATA_SCRIPT)' ] && \
	   [ '$(DATA_STAMP)' -nt '$(ROOT)/tools/gdds_utils.py' ] && \
	   [ '$(DATA_STAMP)' -nt '$(ROOT)/tools/tok_utils.py' ]; then \
		echo "[data] up to date $(DATA_DIR)"; \
	else \
		$(PYTHON) $(DATA_SCRIPT) $(DATA_ARGS); \
		printf '%s\n' "$$args" > '$(DATA_ARGS_FILE)'; \
		touch '$(DATA_STAMP)'; \
	fi

$(TRAIN_BIN): main.c gpt_lm_shared.h $(SHARED_SRC) $(GD_PROGRESS_DEPS) $(GD_EXAMPLE_CONFIG_DEPS) $(LIB) $(BUILD_ARGS_FILE)
	$(CC) $(CPPFLAGS) $(CFLAGS) main.c $(SHARED_SRC) $(GD_PROGRESS_SRC) $(GD_EXAMPLE_CONFIG_SRC) $(LIB) $(LDLIBS) -o $@

$(INFER_BIN): infer.c gpt_lm_shared.h $(SHARED_SRC) $(LIB) $(BUILD_ARGS_FILE)
	$(CC) $(CPPFLAGS) $(CFLAGS) infer.c $(SHARED_SRC) $(LIB) $(LDLIBS) -o $@

$(GENERATE_BIN): main_generate.c gpt_lm_shared.h $(SHARED_SRC) $(LIB) $(BUILD_ARGS_FILE)
	$(CC) $(CPPFLAGS) $(CFLAGS) main_generate.c $(SHARED_SRC) $(LIB) $(LDLIBS) -o $@

run: $(TRAIN_BIN) $(DATA_STAMP) $(CONFIG)
	GRADIENTS_METALLIB=$(METALLIB) ./$(TRAIN_BIN) --config $(CONFIG)

infer: $(INFER_BIN) $(DATA_STAMP)
	GRADIENTS_METALLIB=$(METALLIB) ./$(INFER_BIN) --data-dir $(DATA_DIR) $(ARGS)

interactive: $(GENERATE_BIN) $(DATA_STAMP)
	GRADIENTS_METALLIB=$(METALLIB) ./$(GENERATE_BIN) --data-dir $(DATA_DIR) $(ARGS)

overfit-smoke: $(BIN) $(DATA_STAMP) configs/overfit_smoke.yaml
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --config configs/overfit_smoke.yaml

generate-smoke: $(BIN) $(DATA_STAMP) configs/generate_smoke.yaml
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --config configs/generate_smoke.yaml

periodic-generate-smoke: $(BIN) $(DATA_STAMP) configs/periodic_generate_smoke.yaml
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --config configs/periodic_generate_smoke.yaml

checkpoint-smoke: $(TRAIN_BIN) $(INFER_BIN) $(DATA_STAMP) configs/checkpoint_smoke.yaml
	GRADIENTS_METALLIB=$(METALLIB) ./$(TRAIN_BIN) --config configs/checkpoint_smoke.yaml
	GRADIENTS_METALLIB=$(METALLIB) ./$(INFER_BIN) --data-dir $(DATA_DIR) --checkpoint checkpoints/smoke_best.gdckpt --prompt "Don Abbondio" --max-new-tokens 8

smoke: overfit-smoke generate-smoke periodic-generate-smoke

clean:
	rm -f $(TRAIN_BIN) $(INFER_BIN) $(GENERATE_BIN) $(BUILD_ARGS_FILE)
	rm -rf $(DATA_DIR) checkpoints
