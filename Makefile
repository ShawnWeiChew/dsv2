CC=gcc
CFLAGS=-Wall
DEBUG_CFLAGS=-fsanitize=address -g -fno-omit-frame-pointer
INCLUDE=-Iinclude -Ithirdparty
LDFLAGS=-lm

TARGET_FOLDER=build
TARGET_NAME=$(TARGET_FOLDER)/deepseek
TOKENIZER_TEST=$(TARGET_FOLDER)/test_tokenizer

SRC_FOLDER=src
TARGET_FILE_NAME=$(SRC_FOLDER)/deepseek.c
TARGET_OBJ_FILE=$(TARGET_FOLDER)/deepseek.o

SRC_FILES=$(shell find $(SRC_FOLDER) -type f -name '*.c' ! -wholename '$(TARGET_FILE_NAME)')
SRC_OBJ_FILES=$(patsubst $(SRC_FOLDER)/%.c, $(TARGET_FOLDER)/%.o, $(SRC_FILES))

# test targets
TEST_TOKENIZER_TARGET=$(TARGET_FOLDER)/test_tokenizer
TEST_OPS_TARGET=$(TARGET_FOLDER)/test_ops

.PHONY: clean check all

all : $(TARGET_NAME)

$(TARGET_NAME) : $(TARGET_OBJ_FILE) $(SRC_OBJ_FILES)
	$(CC) $(CFLAGS) $(INCLUDE) $(LDFLAGS) -o $@ $^

$(TARGET_OBJ_FILE) : $(TARGET_FILE_NAME) | $(TARGET_FOLDER)
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $^

$(SRC_OBJ_FILES) : $(SRC_FILES) | $(TARGET_FOLDER)
	$(CC) $(CFLAGS) $(INCLUDE) $(LDFLAGS) -c -o $@ $<

check : $(TEST_OPS_TARGET)
	./build/test_ops
# 	./build/test_tokenizer

$(TEST_TOKENIZER_TARGET) : tests/test_tokenizer.c src/tokenizer.c | $(TARGET_FOLDER)
	$(CC) $(INCLUDE) $(DEBUG_CFLAGS) -o $@ $^ 

$(TEST_OPS_TARGET) : tests/test_ops.c src/ops.c | $(TARGET_FOLDER)
	$(CC) $(INCLUDE) $(DEBUG_CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_FOLDER) :
	mkdir -p $(TARGET_FOLDER)

clean:
	rm -rf build