#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "../thirdparty/json.h"
#include "config.h"
#include <stdbool.h>
#include <unistd.h>

#define DS_TOKENIZER_FILE_NAME       "tokenizer.json"
#define DS_TOKENIZER_BOS             100000
#define DS_TOKENIZER_EOS             100001
#define DS_SPACE_CHARCTER_IDX        207
#define DS_SPACE_CHARACTER_UTF_IDX   0xA0C4
#define DS_NEWLINE_CHARACTER_UTF_IDX 0x8AC4

extern DeepseekConfig deepseek_config;

typedef struct {
    char token_combination[DS_MAX_TOKEN_LEN * 2 + 1];
} TokenizerMergeTarget; //  index in vocab is idx + DS_MERGE_PAIR_OFFSET

typedef struct {
    char **vocab;
    TokenizerMergeTarget *merge_targets;
} Tokenizer;

void build_tokenizer(Tokenizer *t);

void encode(Tokenizer *t, char *input_sequence, int *token_buffer, size_t *token_buffer_len);

bool decode(Tokenizer *t, int output_logit, char *output_token_buffer);

void free_tokenizer(Tokenizer *t);

#endif