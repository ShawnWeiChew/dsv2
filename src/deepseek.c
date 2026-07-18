#include "../include/config.h"
#include "../include/ds_state.h"
#include "../include/ops.h"
#include "../include/tokenizer.h"
#include <stdio.h>
#include <time.h>

DeepseekConfig deepseek_config = {
    .vocab_size = DS_VOCAB_SIZE,
    .hidden_dim = DS_HIDDEN_DIM,
    .first_k_dense_replace = DS_FIRST_K_DENSE,
    .max_sequence_len = DS_MAX_SEQ_LEN,
    .n_layers = DS_N_LAYERS,
    .moe_hidden_size = DS_MOE_HIDDEN_SIZE,
    .n_routed_experts = DS_N_ROUTED_EXPERTS,
    .n_shared_experts = DS_N_SHARED_EXPERTS,
    .n_experts_per_token = DS_N_EXPERTS_PER_TOK,
    .n_attn_heads = DS_N_ATTN_HEADS,
    .kv_lora_rank = DS_KV_LORA_RANK,
    .qk_nope_head_dim = DS_QK_NOPE_HEAD_DIM,
    .qk_rope_head_dim = DS_QK_ROPE_HEAD_DIM,
    .rms_norm_eps = DS_RMS_NORM_EPS,
};

int main() {
    // Tokenizer tokenizer;
    // build_tokenizer(&tokenizer);

    // char t1[] = "I love Deepseek!";
    // int output_buffer[128];
    // size_t output_buffer_len;
    // encode(&tokenizer, t1, output_buffer, &output_buffer_len);

    // for (int i = 0; i < output_buffer_len; i++) {
    //     printf("%d | ", output_buffer[i]);
    // }
    // printf("\n");

    // free_tokenizer(&tokenizer);

    // YarnConstants yc = {.cos = NULL, .sin = NULL};
    // setup_yarn_sin_cos_cache(&deepseek_config, &yc, 2048);
    // yarn(NULL, NULL, 1, 1, &yc);

    DSWeights weights;
    load_weights(&weights);
}