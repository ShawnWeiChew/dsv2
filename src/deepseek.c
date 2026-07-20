#include "../include/config.h"
#include "../include/ds_state.h"
#include "../include/ops.h"
#include "../include/tokenizer.h"
#include <alloca.h>
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
    .mlp_hidden = DS_MLP_HIDDEN,
};

char prompt[] = "Argentina will win the world cup because ";
const int max_generation_length = 2048;

int main() {
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);

    int input_token_buffer[128];
    size_t input_token_buffer_len;

    encode(&tokenizer, prompt, input_token_buffer, &input_token_buffer_len);

    YarnConstants yc = {.cos = NULL, .sin = NULL};
    setup_yarn_sin_cos_cache(&deepseek_config, &yc, max_generation_length);

    DSWeights weights;
    load_weights(&weights);

    DSRunningState state;
    allocate_running_state(&state, &deepseek_config, max_generation_length);

    int token = input_token_buffer[0];
    int next;

    puts("\n\n");

    // main decode loop
    for (int t = 0; t < max_generation_length; t++) {
        // look up the embeddings
        ds_get_quantized_embeddings(&state, &deepseek_config, &weights.embed, token);

        for (int layer_no = 0; layer_no < deepseek_config.n_layers; layer_no++) {
            if (layer_no == 0) {
                ds_dense_decoder_layer(
                    &state,
                    &deepseek_config,
                    &weights,
                    state.token_embedding_scratch,
                    layer_no,
                    t,
                    &yc
                );
            } else {
                float *moe_input = layer_no == 1 ? state.mlp_down_scratch : state.moe_ffn_sum;

                ds_moe_decode_layer(
                    &state, &deepseek_config, &weights, moe_input, layer_no, t, &yc
                );
            }
        }

        // perform the final layernorm
        rms_norm(
            state.moe_ffn_sum,
            weights.model_layernorm,
            1,
            deepseek_config.hidden_dim,
            deepseek_config.rms_norm_eps
        );

        // lm_head
        ds_matmul_4bit(
            state.lm_head_scratch,
            state.moe_ffn_sum,
            weights.lm_head.lm_head_weights,
            weights.lm_head.lm_head_scales,
            weights.lm_head.lm_head_biases,
            deepseek_config.vocab_size,
            deepseek_config.hidden_dim
        );

        // based on the result, argmax the top token if we are in the decode phase
        // the results are also appended to the input token_buffer
        if (t >= input_token_buffer_len - 1) {
            next = ds_sample_argmax(&state, &deepseek_config);

            // NOTE: I think this part was done a little poorly by me, but the logic should just be
            // as such we are looking at tokens 0..t to predict token t + 1 that is the whole setup
            input_token_buffer[t + 1] = next;

            // print the generated result
            char output_token_buffer[DS_MAX_TOKEN_LEN];
            decode(&tokenizer, next, output_token_buffer);
            printf("%s", output_token_buffer);
            fflush(stdout);
        } else {
            next = input_token_buffer[t + 1];
        }

        token = next;
    }

    free_running_state(&state);
    free_yarn_sin_cos_cache(&yc);
    free_weights(&weights);
    free_tokenizer(&tokenizer);
}