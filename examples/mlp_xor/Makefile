ROOT := ../..
BUILD_DIR := $(ROOT)/build
LIB := $(BUILD_DIR)/libgradients.a
METALLIB := $(BUILD_DIR)/gradients.metallib
BIN := gd_main_mlp_xor
DATASET := data/xor-00000.gdds

CC ?= cc
PYTHON ?= python3
UNAME_S := $(shell uname -s)
CPPFLAGS := -I$(ROOT)/include
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS := -pthread

ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework Foundation -framework Metal
endif

.PHONY: all run build-lib clean

all: $(BIN)

build-lib:
	$(MAKE) -C $(ROOT) build

$(DATASET): dataset.py $(ROOT)/tools/gdds_utils.py
	$(PYTHON) dataset.py --out-dir data --split xor

$(BIN): main.c build-lib $(DATASET)
	$(CC) $(CPPFLAGS) $(CFLAGS) main.c $(LIB) $(LDLIBS) -o $@

run: $(BIN)
	GRADIENTS_METALLIB=$(METALLIB) ./$(BIN)

clean:
	rm -f $(BIN)
	rm -rf data
