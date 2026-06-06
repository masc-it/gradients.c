ROOT := ../..
BUILD_DIR := $(ROOT)/build
LIB := $(BUILD_DIR)/libgradients.a
METALLIB := $(BUILD_DIR)/gradients.metallib
BIN := gd_main_mlp_mnist
DATA_DIR ?= data
DATA_STAMP := $(DATA_DIR)/.mnist_gdds.stamp
DATA_ARGS_FILE := $(DATA_DIR)/.mnist_gdds.args

CC ?= cc
PYTHON ?= python3
UNAME_S := $(shell uname -s)
CPPFLAGS := -I$(ROOT)/include
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS := -pthread

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

build-lib:
	$(MAKE) -C $(ROOT) build

data: $(DATA_STAMP)

$(DATA_STAMP): FORCE dataset.py $(ROOT)/tools/gdds_utils.py
	@mkdir -p $(DATA_DIR)
	@args='$(DATA_ARGS)'; \
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
	GRADIENTS_METALLIB=$(METALLIB) GD_MNIST_DATA_DIR=$(DATA_DIR) ./$(BIN)

smoke:
	$(MAKE) clean DATA_DIR=data_smoke
	GD_MNIST_EPOCHS=1 GD_MNIST_REPORT_EVERY=5 \
		$(MAKE) run DATA_DIR=data_smoke MNIST_TRAIN_LIMIT=4096 MNIST_TEST_LIMIT=1000

clean:
	rm -f $(BIN)
	rm -rf $(DATA_DIR)
ifneq ($(DATA_DIR),data_smoke)
	rm -rf data_smoke
endif
