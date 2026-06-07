ROOT := ../..
BUILD_DIR := $(ROOT)/build
LIB := $(BUILD_DIR)/libgradients.a
METALLIB := $(BUILD_DIR)/gradients.metallib
BIN := gd_main_gpt_lm
DATA_DIR ?= data
DATA_STAMP := $(DATA_DIR)/.gpt_lm_gdds.stamp
DATA_ARGS_FILE := $(DATA_DIR)/.gpt_lm_gdds.args
DATA_FORMAT_VERSION := gpt-lm-promessi-v1

CC ?= cc
PYTHON ?= uv run --with numpy
UNAME_S := $(shell uname -s)
CPPFLAGS := -I$(ROOT)/include
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic
LDLIBS := -pthread -lm

ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework Foundation -framework Metal
endif

DATA_ARGS := --out-dir $(DATA_DIR) --preview-samples 0
ifneq ($(GPT_LM_SOURCE),)
DATA_ARGS += --source $(GPT_LM_SOURCE)
endif
ifneq ($(GPT_LM_VOCAB_SIZE),)
DATA_ARGS += --vocab-size $(GPT_LM_VOCAB_SIZE)
endif
ifneq ($(GPT_LM_CONTEXT_LENGTH),)
DATA_ARGS += --context-length $(GPT_LM_CONTEXT_LENGTH)
endif
ifneq ($(GPT_LM_MAX_SHARD_BYTES),)
DATA_ARGS += --max-shard-bytes $(GPT_LM_MAX_SHARD_BYTES)
endif
ifneq ($(GPT_LM_RETRAIN_TOKENIZER),)
DATA_ARGS += --retrain-tokenizer
endif

.PHONY: all run build-lib tools data clean smoke overfit-smoke generate-smoke FORCE

all: $(BIN)

build-lib:
	$(MAKE) -C $(ROOT) build

tools:
	$(MAKE) -C $(ROOT) tools

data: $(DATA_STAMP)

$(DATA_STAMP): FORCE dataset.py $(ROOT)/tools/gdds_utils.py $(ROOT)/tools/tok_utils.py tools
	@mkdir -p $(DATA_DIR)
	@args='$(DATA_FORMAT_VERSION) $(DATA_ARGS)'; \
	if [ -f '$(DATA_STAMP)' ] && [ -f '$(DATA_ARGS_FILE)' ] && \
	   [ "$$(cat '$(DATA_ARGS_FILE)')" = "$$args" ]; then \
		echo "[data] up to date $(DATA_DIR)"; \
	else \
		$(PYTHON) dataset.py $(DATA_ARGS); \
		printf '%s\n' "$$args" > '$(DATA_ARGS_FILE)'; \
		touch '$(DATA_STAMP)'; \
	fi

$(BIN): main.c build-lib $(DATA_STAMP)
	$(CC) $(CPPFLAGS) $(CFLAGS) main.c $(LIB) $(LDLIBS) -o $@

run: $(BIN)
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --data-dir $(DATA_DIR) $(ARGS)

overfit-smoke: $(BIN)
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --data-dir $(DATA_DIR) --epochs 1 --batch-size 1 --overfit-num-samples 1 --report-every 1 $(ARGS)

generate-smoke: $(BIN)
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --data-dir $(DATA_DIR) --epochs 0 --generate "Don Abbondio" --max-new-tokens 16 $(ARGS)

smoke: overfit-smoke generate-smoke

clean:
	rm -f $(BIN)
	rm -rf $(DATA_DIR)
