#include "../include/ops.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void softmax(float *in, size_t M, size_t N) {
    for (int i = 0; i < M; i++) {
        // first find the running sum and the max
        float max = -1E20;
        float running_sum = 0;
        int outer_idx = i * N;

        for (int j = 0; j < N; j++) {
            float new_max = in[outer_idx + j] > max ? in[outer_idx + j] : max;

            float correction_factor = expf(max - new_max);
            running_sum *= correction_factor;

            max = new_max;
            running_sum += expf(in[outer_idx + j] - max);
        }

        for (int j = 0; j < N; j++) {
            in[outer_idx + j] = expf(in[outer_idx + j] - max) / running_sum;
        }
    }
}

void rms_norm(float *in, _Float16 *element_wise_affine, size_t M, size_t N, float eps) {
    for (int i = 0; i < M; i++) {
        // first collect the sum
        int outer_idx = i * N;
        float rms_sum = 0;

        for (int j = 0; j < N; j++) {
            rms_sum += in[outer_idx + j] * in[outer_idx + j];
        }

        rms_sum = sqrtf(rms_sum / N + eps);

        for (int j = 0; j < N; j++) {
            in[outer_idx + j] = in[outer_idx + j] / rms_sum * element_wise_affine[j];
        }
    }
}

void silu(float *in, size_t M, size_t N) {
    for (int i = 0; i < M; i++) {
        int outer_idx = i * N;
        for (int j = 0; j < N; j++) {
            in[outer_idx + j] = in[outer_idx + j] * (float)1 / (1 + expf(-in[outer_idx + j]));
        }
    }
}

// NOTE to self: this should have been INT4, not float
static inline void unpack_weights(uint32_t weights, uint8_t *out) {
    for (int j = 0; j < DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32; j++) {
        uint8_t val = (weights >> (j * 4) & 0b1111);
        memcpy(&out[j], &val, sizeof(uint8_t));
    }
}

void ds_matmul_4bit(
    float *out, float *in, uint32_t *weights, _Float16 *scales, _Float16 *biases, size_t N, size_t K
) {
    // target each output row
    int weight_idx = 0;     // increments every 8 increments of in_idx
    int scale_bias_idx = 0; // increments every 8 increments of weight_idx

    for (int row = 0; row < N; row++) {
        // TODO: there is an optimization here where I can do vector loads instead
        int in_idx = 0;
        float sum = 0;

        while (in_idx < K) {
            uint8_t expanded_weights[DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32] = {};
            // first take out the weights

            unpack_weights(weights[weight_idx], expanded_weights);

            // then take out the scales
            _Float16 scale = scales[scale_bias_idx];
            _Float16 bias = biases[scale_bias_idx];

            for (int i = 0; i < DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32; i++) {
                // NOTE: MLX does quantization with w*S + bias, not the traditional (w - bias) * S
                // :cries:
                sum += ((float)expanded_weights[i] * scale + bias) * in[in_idx + i];
            }

            in_idx += DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32;
            weight_idx++;
            if (weight_idx % 8 == 0) {
                scale_bias_idx++;
            }
        }

        out[row] = sum;
    }
}

void matmul(float *c, float *a, _Float16 *b, size_t N, size_t K) {
    for (int i = 0; i < N; i++) {
        int outer_idx = i * K;

        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += b[outer_idx + j] * a[j];
        }
        c[i] = sum;
    }
}

void setup_yarn_sin_cos_cache(
    DeepseekConfig *config, YarnConstants *yarn_constants, size_t cache_length
) {
    // NOTE: this value can probably be a little smaller, since I dont think this is ever going to
    // run till the max sequence length
    // give the option to resize and recompute the cache if the sequence length overflows
    if (yarn_constants->cos == NULL || yarn_constants->sin == NULL) {
        yarn_constants->cos = calloc(cache_length * config->qk_rope_head_dim, sizeof(float));
        yarn_constants->sin = calloc(cache_length * config->qk_rope_head_dim, sizeof(float));
        yarn_constants->access_pattern =
            calloc(config->qk_rope_head_dim, sizeof(YarnInterleavedAccessPattern));
    } else {
        yarn_constants->cos =
            realloc(yarn_constants->cos, cache_length * config->qk_rope_head_dim * sizeof(float));
        yarn_constants->sin =
            realloc(yarn_constants->sin, cache_length * config->qk_rope_head_dim * sizeof(float));
    }

    if (yarn_constants->cos == NULL || yarn_constants->sin == NULL ||
        yarn_constants->access_pattern == NULL) {
        perror("Could not allocate space for yarn buffers");
        exit(1);
    }

    // first, find the range for which the smooth rotation is applied
    int lowest_dim = (float)32 *
                     logf(DS_INTIAL_CONTEXT_LEN / (2 * M_PI * DS_YARN_ROTATION_CEILING)) * 1 /
                     logf(DS_YARN_BASE);

    int highest_dim = ceil(
        (float)config->qk_rope_head_dim / 2 *
        logf(DS_INTIAL_CONTEXT_LEN / (2 * M_PI * DS_YARN_ROTATION_FLOOR)) * 1 / logf(DS_YARN_BASE)
    );

    float rotation_freqs[config->qk_rope_head_dim / 2];
    for (int i = 0; i < config->qk_rope_head_dim; i += 2) {
        float freq_original = 1 / powf(DS_YARN_BASE, (float)i / config->qk_rope_head_dim);
        float freq_scaled = freq_original / DS_SCALE_FACTOR;

        float clamped_value = 1;
        if (i / 2 >= lowest_dim && i / 2 <= highest_dim) {
            clamped_value = ((float)i / 2 - lowest_dim) / (highest_dim - lowest_dim);
        } else if (i / 2 < lowest_dim) {
            clamped_value = 0;
        }

        rotation_freqs[i / 2] = clamped_value * freq_scaled + (1 - clamped_value) * freq_original;
    }

    for (int j = 0; j < cache_length; j++) {
        float *current_cos_cache_position = yarn_constants->cos + (j * config->qk_rope_head_dim);
        float *current_sin_cache_position = yarn_constants->sin + (j * config->qk_rope_head_dim);

        for (int i = 0; i < config->qk_rope_head_dim / 2; i++) {
            // apply scaled position
            current_cos_cache_position[i] = cosf(j * rotation_freqs[i]);
            current_cos_cache_position[i + config->qk_rope_head_dim / 2] =
                cosf(j * rotation_freqs[i]);

            current_sin_cache_position[i] = sinf(j * rotation_freqs[i]);
            current_sin_cache_position[i + config->qk_rope_head_dim / 2] =
                sinf(j * rotation_freqs[i]);
        }
    }

    // setup access pattern -- this is kind of jank, idk if there is a better solution
    for (int i = 0; i < config->qk_rope_head_dim; i++) {
        bool is_before_halfway = i < config->qk_rope_head_dim / 2;
        yarn_constants->access_pattern[i].idx1 =
            is_before_halfway ? i * 2 : (i - config->qk_rope_head_dim / 2) * 2 + 1;

        yarn_constants->access_pattern[i].idx2 =
            is_before_halfway ? i * 2 + 1 : (i - config->qk_rope_head_dim / 2) * 2;
        yarn_constants->access_pattern[i].idx_2_neg = is_before_halfway;
    }
}

void yarn(float *in, float *out, size_t M, size_t N, YarnConstants *yarn_constants) {
    for (int i = 0; i < M; i++) {
        int outer_idx = i * N;

        for (int j = 0; j < N; j++) {
            out[outer_idx + j] = in[outer_idx + yarn_constants->access_pattern[j].idx1] *
                                     yarn_constants->cos[outer_idx + j] +
                                 in[outer_idx + yarn_constants->access_pattern[j].idx2] *
                                     yarn_constants->sin[outer_idx + j] *
                                     (yarn_constants->access_pattern[j].idx_2_neg ? -1 : 1);
        }
    }
}

void free_yarn_sin_cos_cache(YarnConstants *yarn_constants) {
    free(yarn_constants->sin);
    free(yarn_constants->cos);
    free(yarn_constants->access_pattern);
}

void transpose(float *in, float *out, size_t M, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out[i * M + j] = in[j * N + i];
        }
    }
}

static int compare_expert_scores(const void *a, const void *b) {
    // qsort itself returns < 0 if the first should come before second
    // we want the highest scores first, so we should return the negative value

    // NOTE: this was a simple thing to do but I kind of missed it
    if (((DSRoutedExpert *)a)->score > ((DSRoutedExpert *)b)->score) {
        return -1;
    } else if (((DSRoutedExpert *)a)->score < ((DSRoutedExpert *)b)->score) {
        return 1;
    }
    return 0;
}

// we have the small advantage that there is no need for sorting within the experts themselves
// there is probably some better implementation of this using a min heap, but I am going to make
// this as simply as possible first
void identify_topk(
    DSRoutedExpert top_experts[], float *gating_result, int num_experts_to_select,
    int num_experts_total
) {
    DSRoutedExpert all_experts[num_experts_total];

    for (int i = 0; i < num_experts_total; i++) {
        all_experts[i].score = gating_result[i];
        all_experts[i].idx = i;
    }

    qsort(all_experts, num_experts_total, sizeof(DSRoutedExpert), compare_expert_scores);
    memcpy(top_experts, all_experts, sizeof(DSRoutedExpert) * num_experts_to_select);
}

// put result into silu result
static void compute_swiglu(float *silu_result, float *swiglu_activation, size_t M, size_t N) {
    for (int i = 0; i < M; i++) {
        int outer_idx = i * N;
        for (int j = 0; j < N; j++) {
            silu_result[outer_idx + j] =
                silu_result[outer_idx + j] * swiglu_activation[outer_idx + j];
        }
    }
}

void ds_moe_layer(
    DSRunningState *state, DeepseekConfig *config, DSMoELayerWeights *weights, float *in
) {
    // input is (2048, 1) -> (64, 1)
    matmul(
        state->topk_routing_results,
        in,
        weights->moe_gate_weights,
        config->n_routed_experts,
        config->hidden_dim
    );

    // NOTE: we currently assume that every input has sequence length of 1
    softmax(state->topk_routing_results, 1, config->n_routed_experts);

    DSRoutedExpert top_experts[config->n_experts_per_token];
    identify_topk(
        top_experts,
        state->topk_routing_results,
        config->n_experts_per_token,
        config->n_routed_experts
    );

    memset(state->moe_ffn_sum, 0, sizeof(float) * config->hidden_dim);

    // perform expert routing and ffn
    for (int i = 0; i < config->n_experts_per_token; i++) {
        int scales_and_biases_up_offset = top_experts[i].idx * config->moe_hidden_size * 32;
        int weights_up_offset = top_experts[i].idx * config->moe_hidden_size * 256;

        // (1408, 2048) @ (2048,) -> (1408,)
        // oddly enough, this up proj is smaller
        ds_matmul_4bit(
            state->routed_expert_up_scratch,
            in,
            weights->routed_up_weights + weights_up_offset,
            weights->routed_up_scales + scales_and_biases_up_offset,
            weights->routed_up_biases + scales_and_biases_up_offset,
            config->moe_hidden_size,
            config->hidden_dim
        );

        // other swiglu weight
        // (1408, 2048) @ (2048,) -> (1408,)
        ds_matmul_4bit(
            state->routed_expert_swiglu_scratch,
            in,
            weights->routed_gate_weights + weights_up_offset,
            weights->routed_gate_scales + scales_and_biases_up_offset,
            weights->routed_gate_biases + scales_and_biases_up_offset,
            config->moe_hidden_size,
            config->hidden_dim
        );

        silu(state->routed_expert_swiglu_scratch, 1, config->moe_hidden_size);

        compute_swiglu(
            state->routed_expert_up_scratch,
            state->routed_expert_swiglu_scratch,
            1,
            config->moe_hidden_size
        );

        // down proj
        // (2048, 1408) @ (1408,) -> (2048,)
        int scales_and_biases_down_offset = top_experts[i].idx * config->hidden_dim * 22;
        int weights_down_offset = top_experts[i].idx * config->hidden_dim * 176;
        ds_matmul_4bit(
            state->routed_expert_down_scratch,
            state->routed_expert_up_scratch,
            weights->routed_down_weights + weights_down_offset,
            weights->routed_down_scales + scales_and_biases_down_offset,
            weights->routed_down_biases + scales_and_biases_down_offset,
            config->hidden_dim,
            config->moe_hidden_size
        );

        // apply expert scaling
        float current_expert_score = top_experts[i].score;
        for (int j = 0; j < config->hidden_dim; j++) {
            state->moe_ffn_sum[j] += current_expert_score * state->routed_expert_down_scratch[j];
        }
    }

    // shared expert ffn
    // the weights format combines tow of them together so we just apply the weights at the same
    // time
    // (1408 * 2, 2048) @ (2048,) -> (1408 * 2,)
    ds_matmul_4bit(
        state->shared_expert_up_scratch,
        in,
        weights->shared_up_weights,
        weights->shared_up_scales,
        weights->shared_up_biases,
        config->moe_hidden_size * 2,
        config->hidden_dim
    );

    // (1408 * 2, 2048) @ (2048,) -> (1408 * 2,)
    ds_matmul_4bit(
        state->shared_expert_swiglu_scratch,
        in,
        weights->shared_gate_weights,
        weights->shared_gate_scales,
        weights->shared_gate_biases,
        config->moe_hidden_size * 2,
        config->hidden_dim
    );

    silu(state->shared_expert_swiglu_scratch, 1, config->moe_hidden_size * 2);

    compute_swiglu(
        state->shared_expert_up_scratch,
        state->shared_expert_swiglu_scratch,
        1,
        config->moe_hidden_size * 2
    );

    // (2048, 1408 * 2) @ (1408 * 2,) -> (2048,)
    ds_matmul_4bit(
        state->shared_expert_down_scratch,
        state->shared_expert_up_scratch,
        weights->shared_down_weights,
        weights->shared_down_scales,
        weights->shared_down_biases,
        config->hidden_dim,
        config->moe_hidden_size * 2
    );

    for (int i = 0; i < config->hidden_dim; i++) {
        state->moe_ffn_sum[i] += state->shared_expert_down_scratch[i];
    }
}

static void ds_mla_naive_qk_attention_score_matmul(
    DeepseekConfig *config, float *k_nope, float *k_rope, float *q_nope, float *q_rope,
    int layer_no, int seq_no, size_t max_generation_length, float *qk_attention_scores_scratch,
    float *final_attention_scores
) {
    memset(final_attention_scores, 0, config->hidden_dim * sizeof(float));

    for (int h = 0; h < config->n_attn_heads; h++) {
        // calculate like this because the offsets for the rope heads are 192 apart
        // there is no sequence based offset since there is only 1 query
        int head_offset_for_q_nope = h * (config->qk_rope_head_dim + config->qk_nope_head_dim);

        // iterate through the sequence
        // NOTE: this is something I glossed over in llama2.c, but attention requires that you
        // compute the attention scores for the current token as well.
        for (int s = 0; s <= seq_no; s++) {
            float sequence_sum = 0;

            int sequence_offset_for_k_nope =
                layer_no * max_generation_length * (config->hidden_dim * 2) + // layer offset
                s * (config->hidden_dim * 2) +                                // sequence offset
                h * (config->qk_nope_head_dim * 2);                           // head offset

            // iterate through the dimensions
            // (H, S, 128) @ (H, S, 128)
            for (int j = 0; j < config->qk_nope_head_dim; j++) {
                sequence_sum +=
                    q_nope[head_offset_for_q_nope + j] * k_nope[sequence_offset_for_k_nope + j];
            }

            // and then the rope
            int q_rope_offset = h * config->qk_rope_head_dim;
            // no need to take into account head position, since this is broadcasted among the heads
            int k_rope_offset = layer_no * max_generation_length * config->qk_rope_head_dim +
                                s * config->qk_rope_head_dim;
            for (int j = 0; j < config->qk_rope_head_dim; j++) {
                sequence_sum += q_rope[q_rope_offset + j] * k_rope[k_rope_offset + j];
            }

            // (H, 1, S)
            // the paper's formula uses another factor but it does not seem to be used in the MLX
            // formula should be careful about the correctness of this later
            // NOTE: once again forgot that the sqrt should come before the softmax
            qk_attention_scores_scratch[s] = sequence_sum * sqrtf(config->n_attn_heads);
        }

        // attention softmax can just be computed for every head in the sequence
        // (H, 1, S)
        softmax(qk_attention_scores_scratch, 1, seq_no + 1);

        // then multiply it by the value vector
        // (H, 1, S) @ (H, S, 128)
        int final_attention_score_offset = h * config->qk_nope_head_dim;

        // slightly sketchy matmul, should check again
        for (int s = 0; s <= seq_no; s++) {
            int sequence_offset_for_k_nope =
                layer_no * max_generation_length * (config->hidden_dim * 2) + // layer offset
                s * (config->hidden_dim * 2) +                                // sequence offset
                h * (config->qk_nope_head_dim * 2) +                          // head offset
                config->hidden_dim;                                           //  k in front

            int attn_score_offset = h * config->qk_nope_head_dim;
            for (int i = 0; i < config->qk_nope_head_dim; i++) {
                final_attention_scores[final_attention_score_offset + i] +=
                    qk_attention_scores_scratch[s] * k_nope[sequence_offset_for_k_nope + i];
            }
        }
    }
}

// TODO: verify dimensions, then test
void ds_mla_layer_naive(
    DSRunningState *state, DeepseekConfig *config, DSMLALayerWeights *weights, float *in,
    int layer_no, int seq_no, YarnConstants *yarn_constants
) {

    // generate kv_lora, k_rope
    // (2048) @ (2048, 576) -> (1, 576)
    ds_matmul_4bit(
        state->kv_lora_rope_scratch,
        in,
        weights->kv_a_proj_weights,
        weights->kv_a_proj_scales,
        weights->kv_a_proj_biases,
        (config->kv_lora_rank + config->qk_rope_head_dim),
        config->hidden_dim
    );

    rms_norm(
        state->kv_lora_rope_scratch,
        weights->kv_a_layernorm,
        1,
        config->kv_lora_rank,
        config->rms_norm_eps
    );

    // apply YaRN to rope sequence before caching k rope
    yarn(
        state->kv_lora_rope_scratch + config->kv_lora_rank,
        state->k_rope_cache + layer_no * state->max_sequence_len * config->qk_rope_head_dim +
            seq_no * config->qk_rope_head_dim,
        1,
        config->qk_rope_head_dim,
        yarn_constants
    );

    // generate q_nope and q_rope
    // (2048) @ (2048, 3072) -> (1, 3072) -> (1, 16, 192)
    ds_matmul_4bit(
        state->q_nope_rope_scratch,
        in,
        weights->q_proj_weights,
        weights->q_proj_scales,
        weights->q_proj_biases,
        config->n_attn_heads * (config->qk_nope_head_dim + config->qk_rope_head_dim),
        config->hidden_dim
    );
    // apply rope to every head
    for (int i = 0; i < config->n_attn_heads; i++) {
        int rope_offset = (i + 1) * config->qk_nope_head_dim + i * config->qk_rope_head_dim;
        yarn(
            state->q_nope_rope_scratch + rope_offset,
            state->q_rope_scratch + i * config->qk_rope_head_dim,
            1,
            config->qk_rope_head_dim,
            yarn_constants
        );
    }

    // NOTE: design used in the reference folder caches the up projection instead of the
    // comporessed projection (1, 512) @ (512, 4096) -> (4096) [cached]
    ds_matmul_4bit(
        state->kv_cache + layer_no * state->max_sequence_len * (config->hidden_dim * 2) +
            seq_no * config->hidden_dim * 2,
        state->kv_lora_rope_scratch,
        weights->kv_b_proj_weights,
        weights->kv_b_proj_scales,
        weights->kv_b_proj_biases,
        config->hidden_dim * 2,
        config->kv_lora_rank
    );

    ds_mla_naive_qk_attention_score_matmul(
        config,
        state->kv_cache,
        state->k_rope_cache,
        state->q_nope_rope_scratch,
        state->q_rope_scratch,
        layer_no,
        seq_no,
        state->max_sequence_len,
        state->qk_attention_scores_scratch,
        state->final_attention_score
    );

    // final out proj
    ds_matmul_4bit(
        state->mla_out_proj_res,
        state->final_attention_score,
        weights->o_proj_weights,
        weights->o_proj_scales,
        weights->o_proj_biases,
        config->hidden_dim,
        config->hidden_dim
    );
}

void ds_mlp_layer(
    DSRunningState *state, DeepseekConfig *config, DSMLPLayerWeights *weights, float *in
) {
    ds_matmul_4bit(
        state->mlp_up_scratch,
        in,
        weights->up_proj_weights,
        weights->up_proj_scales,
        weights->up_proj_biases,
        config->mlp_hidden,
        config->hidden_dim
    );

    ds_matmul_4bit(
        state->mlp_swiglu_scratch,
        in,
        weights->gate_proj_weights,
        weights->gate_proj_scales,
        weights->gate_proj_biases,
        config->moe_hidden_size,
        config->hidden_dim
    );

    silu(state->mlp_swiglu_scratch, 1, config->moe_hidden_size);
    compute_swiglu(state->mlp_up_scratch, state->mlp_swiglu_scratch, 1, config->moe_hidden_size);

    ds_matmul_4bit(
        state->mlp_down_scratch,
        state->mlp_up_scratch,
        weights->down_proj_weights,
        weights->down_proj_scales,
        weights->down_proj_biases,
        config->hidden_dim,
        config->mlp_hidden
    );
}
