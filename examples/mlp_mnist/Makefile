ROOT := ../..
BIN := gd_main_mlp_mnist
CONFIG ?= config.yaml
DATA_DIR ?= data
DATA_STAMP := $(DATA_DIR)/.mnist_gdds.stamp
DATA_ARGS_FILE := $(DATA_DIR)/.mnist_gdds.args
DATA_FORMAT_VERSION := mnist-u8-v1

CC ?= cc
PYTHON ?= python3
UNAME_S := $(shell uname -s)
CPPFLAGS := -I$(ROOT)/include
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
include ../common.mk
LDLIBS := -pthread -lm

ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework Foundation -framework Metal
endif

DATA_ARGS := --out-dir $(DATA_DIR)
ifneq ($(MNIST_TRAIN_LIMIT),)
DATA_ARGS += --train-limit $(MNIST_TRAIN_LIMIT)
endif
ifneq ($(MNIST_TEST_LIMIT),)
DATA_ARGS += --test-limit $(MNIST_TEST_LIMIT)
endif
ifneq ($(MNIST_MAX_SHARD_BYTES),)
DATA_ARGS += --max-shard-bytes $(MNIST_MAX_SHARD_BYTES)
endif

.PHONY: all run build-lib data clean smoke FORCE

all: $(BIN)

data: $(DATA_STAMP)

$(DATA_STAMP): FORCE dataset.py $(ROOT)/tools/gdds_utils.py
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

$(BIN): main.c $(GD_EXAMPLE_CONFIG_DEPS) $(LIB) $(DATA_STAMP)
	$(CC) $(CPPFLAGS) $(CFLAGS) main.c $(GD_EXAMPLE_CONFIG_SRC) $(LIB) $(LDLIBS) -o $@

run: $(BIN) $(CONFIG)
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN) --config $(CONFIG)

smoke:
	$(MAKE) clean DATA_DIR=data_smoke
	@mkdir -p data_smoke
	@printf '%s\n' \
		'data_dir: data_smoke' \
		'train_epochs: 1' \
		'report_every: 5' \
		'train_batch: 128' \
		'eval_batch: 100' \
		'dataloader_workers: 1' \
		'dataloader_prefetch_factor: 2' \
		'train_seed: 1234' \
		'dropout_seed: 0xd00d5eed12340000' \
		'dropout_p: 0.10' \
		'lr_max: 0.001' \
		'lr_min: 0.0001' \
		'lr_warmup_steps: -1' \
		'weight_decay: 0.0001' \
		'amp_init_scale: 128' \
		'amp_growth_interval: 64' \
		'amp_max_scale: 4096' > data_smoke/config.yaml
	$(MAKE) run DATA_DIR=data_smoke CONFIG=data_smoke/config.yaml MNIST_TRAIN_LIMIT=4096 MNIST_TEST_LIMIT=1000

clean:
	rm -f $(BIN)
	rm -rf $(DATA_DIR)
ifneq ($(DATA_DIR),data_smoke)
	rm -rf data_smoke
endif
