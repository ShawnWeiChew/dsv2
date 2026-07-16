#include "../include/tokenizer.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Helper: look up a key in a json object, returns the value or NULL
static struct json_value_s *json_object_get(struct json_value_s *obj_val, const char *key) {
    struct json_object_s *obj = json_value_as_object(obj_val);
    if (!obj)
        return NULL;

    for (struct json_object_element_s *el = obj->start; el; el = el->next) {
        if (strcmp(el->name->string, key) == 0) {
            return el->value;
        }
    }
    return NULL;
}

void build_tokenizer(Tokenizer *t) {
    t->vocab = (char **)calloc(DS_VOCAB_SIZE, sizeof(char *));
    t->merge_targets =
        (TokenizerMergeTarget *)calloc(DS_N_MERGE_TARGETS, sizeof(TokenizerMergeTarget));

    if (t->vocab == NULL || t->merge_targets == NULL) {
        fprintf(stderr, "could not allocate space for tokenizer");
        exit(1);
    }

    FILE *f = fopen(DS_TOKENIZER_FILE_NAME, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s\n", DS_TOKENIZER_FILE_NAME);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_buf = malloc(file_size);
    if (!json_buf) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    fread(json_buf, 1, file_size, f);
    fclose(f);

    // 2. Parse
    struct json_value_s *root = json_parse(json_buf, file_size);
    free(json_buf); // json.h copies what it needs into its own allocation
    if (!root) {
        fprintf(stderr, "json parse failed\n");
        exit(1);
    }

    // 3. Navigate: root["model"]["vocab"] and root["model"]["merges"]
    struct json_value_s *model = json_object_get(root, "model");
    if (!model) {
        fprintf(stderr, "missing 'model' key\n");
        exit(1);
    }

    struct json_value_s *vocab_val = json_object_get(model, "vocab");
    struct json_value_s *merges_val = json_object_get(model, "merges");

    // vocab is typically an object mapping string -> int
    struct json_object_s *vocab = json_value_as_object(vocab_val);
    if (vocab) {
        // printf("vocab entries: %zu\n", vocab->length);
        // iterate:
        for (struct json_object_element_s *el = vocab->start; el; el = el->next) {
            const char *token = el->name->string;
            struct json_number_s *num = json_value_as_number(el->value);

            assert(num != NULL && token != NULL && "Could not parse the vocab entry");

            int idx = atoi(num->number);

            // allocate the buffer for the string
            size_t token_len = strlen(token);

            // add null terminator
            char *str_buffer = (char *)malloc(token_len + 1);
            strlcpy(str_buffer, token, token_len + 1);

            t->vocab[idx] = str_buffer;
        }
    }

    struct json_array_s *merges = json_value_as_array(merges_val);
    if (merges) {
        // printf("merge entries: %zu\n", merges->length);
        int idx = 0;
        for (struct json_array_element_s *el = merges->start; el; el = el->next) {
            struct json_string_s *rule = json_value_as_string(el->value);

            strlcpy(
                (char *)t->merge_targets[idx].token_combination,
                rule->string,
                DS_MAX_TOKEN_LEN * 2 + 1
            );
            idx++;
        }
    }
}

// NOTE: this is very inefficient but I think thats okay because the sequence is small + we are
// aiming for correctness
static int idx_lookup(Tokenizer *t, char *target) {
    if (strcmp(target, " ") == 0) {
        // this is the hardcoded substitution from space to Ġ
        return DS_SPACE_CHARCTER_IDX;
    }

    for (int i = 0; i < DS_VOCAB_SIZE; i++) {
        if (strcmp(t->vocab[i], target) == 0) {
            return i;
        }
    }

    return -1;
}

static int check_merge_pair(Tokenizer *t, char *merge_candidate) {
    for (int i = 0; i < DS_N_MERGE_TARGETS; i++) {
        if (strcmp(t->merge_targets[i].token_combination, merge_candidate) == 0) {
            return i + DS_MERGE_PAIR_OFFSET_IDX;
        }
    }

    return DS_VOCAB_SIZE + 1;
}

void encode(Tokenizer *t, char *input_sequence, int *token_buffer, size_t *token_buffer_len) {
    // first split the input text into the required tokens we are going to take the simple case
    // and assume that there are only ascii inputs here
    // TODO: account for other uint8 characters

    // add the BOS sequence
    token_buffer[0] = DS_TOKENIZER_BOS;
    (*token_buffer_len)++;

    size_t input_sequence_length = strlen(input_sequence);
    size_t i = 0;
    *token_buffer_len = 1;

    // one for the character, another for the null
    char split_token_buffer[2];

    while (i < input_sequence_length) {
        strlcpy(split_token_buffer, &input_sequence[i++], 2);
        token_buffer[*token_buffer_len] = idx_lookup(t, split_token_buffer);

        (*token_buffer_len)++;
    }

    // after split has happened, merge the tokens
    char merge_buffer[DS_MAX_TOKEN_LEN * 2 + 1];
    while (true) {
        int lowest_idx = DS_VOCAB_SIZE + 1; // the lowest score should have the lowest index
        int best_merge_idx_in_token_buffer = -1;

        // test all merge pairs
        for (int i = 0; i < *token_buffer_len - 1; i++) {
            snprintf(
                merge_buffer,
                DS_MAX_TOKEN_LEN * 2 + 1,
                "%s %s",
                t->vocab[token_buffer[i]],
                t->vocab[token_buffer[i + 1]]
            );
            int merged_idx = check_merge_pair(t, merge_buffer);

            if (merged_idx < lowest_idx) {
                lowest_idx = merged_idx;
                best_merge_idx_in_token_buffer = i;
            }
        }

        if (best_merge_idx_in_token_buffer == -1) {
            break;
        } else {
            token_buffer[best_merge_idx_in_token_buffer] = lowest_idx;

            // copy everything else over
            for (int i = best_merge_idx_in_token_buffer; i < *token_buffer_len - 2; i++) {
                token_buffer[i + 1] = token_buffer[i + 2];
            }

            (*token_buffer_len)--;
        }
    }
}

bool decode(Tokenizer *t, int output_logit, char *output_token_buffer) {
    // false is used to indicate that there is nothing to print
    if (output_logit == DS_TOKENIZER_BOS || output_logit == DS_TOKENIZER_EOS) {
        return false;
    }

    const char *target = t->vocab[output_logit];
    size_t target_size = strlen(target);

    int i = 0;
    int output_token_buffer_idx = 0;

    // hacky way to check for the special Ġ character
    while (i < target_size) {
        // if there is space to look further
        if (i < target_size - 1) {
            uint16_t *char_as_int = (uint16_t *)&target[i];
            if (*char_as_int == DS_SPACE_CHARACTER_UTF_IDX) {
                output_token_buffer[output_token_buffer_idx++] = ' ';
                i += 2;
                continue;
            }
        }

        output_token_buffer[output_token_buffer_idx++] = target[i];
        i++;
    }

    output_token_buffer[output_token_buffer_idx] = '\0';

    return true;
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < DS_VOCAB_SIZE; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->merge_targets);
}