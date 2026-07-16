CC=gcc
CFLAGS=-Wall
INCLUDE=-Iinclude -Ithirdparty

TARGET_FOLDER=build
TARGET_NAME=$(TARGET_FOLDER)/deepseek
TOKENIZER_TEST=$(TARGET_FOLDER)/test_tokenizer

SRC_FOLDER=src
TARGET_FILE_NAME=$(SRC_FOLDER)/deepseek.c
TARGET_OBJ_FILE=$(TARGET_FOLDER)/deepseek.o

SRC_FILES=$(shell find $(SRC_FOLDER) -type f -name '*.c' ! -wholename '$(TARGET_FILE_NAME)')
SRC_OBJ_FILES=$(patsubst $(SRC_FOLDER)/%.c, $(TARGET_FOLDER)/%.o, $(SRC_FILES))


TEST_FOLDER=tests
TEST_FILES=$(shell find $(TEST_FOLDER) -type f -name '*.c')
TEST_OBJ_FILES=$(patsubst $(TEST_FOLDER)/%.c, $(TARGET_FOLDER)/%.o, $(TEST_FILES))

# test targets
TEST_TOKENIZER_TARGET=$(TARGET_FOLDER)/test_tokenizer

.PHONY: clean check all

all : $(TARGET_NAME)

$(TARGET_NAME) : $(TARGET_OBJ_FILE) $(SRC_OBJ_FILES)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^

$(TARGET_OBJ_FILE) : $(TARGET_FILE_NAME) | $(TARGET_FOLDER)
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $^

$(SRC_OBJ_FILES) : $(SRC_FILES) | $(TARGET_FOLDER)
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

check : $(TEST_TOKENIZER_TARGET)
	./build/test_tokenizer

$(TEST_TOKENIZER_TARGET) : build/test_tokenizer.o $(SRC_OBJ_FILES) | $(TARGET_FOLDER)
	$(CC) $(INCLUDE) -o $@ $^

$(TEST_OBJ_FILES) : $(TEST_FILES) | $(TARGET_FOLDER)
	$(CC) $(INCLUDE) -c -o $@ $^

$(TARGET_FOLDER) :
	mkdir -p $(TARGET_FOLDER)

clean:
	rm -rf build