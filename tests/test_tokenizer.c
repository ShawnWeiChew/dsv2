#include "../include/tokenizer.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

bool test_encoding(Tokenizer *t, char *input, int *expected_output, size_t expected_output_len) {
    // NOTE: I made the mistake of allocating too small of a buffer here, which caused it to corrupt
    // the value of expected_output_len
    int token_buffer[strlen(input) + 1];
    size_t output_buf_len;
    encode(t, input, token_buffer, &output_buf_len);

    assert(output_buf_len == expected_output_len && "Output lengths do not match");

    for (int i = 0; i < expected_output_len; i++) {
        if (token_buffer[i] != expected_output[i]) {
            return false;
        }
    }

    for (int i = 0; i < expected_output_len; i++) {
        char print_buf[DS_MAX_TOKEN_LEN + 1];
        if (decode(t, token_buffer[i], print_buf)) {
            printf("%s", print_buf);
        };
    }
    puts(" ");

    return true;
}

int main() {
    Tokenizer t;
    build_tokenizer(&t);

    char t1[] = "I love Deepseek!";
    int expected_output1[] = {100000, 40, 2126, 20593, 30575, 0};
    size_t expected_output1_len = 6;
    assert(test_encoding(&t, t1, expected_output1, expected_output1_len));

    char t2[] = "Work like a dog";
    int expected_output2[] = {100000, 10869, 837, 245, 5025};
    size_t expected_output2_len = 5;
    assert(test_encoding(&t, t2, expected_output2, expected_output2_len));

    puts("All tokenizer tests passed");
}