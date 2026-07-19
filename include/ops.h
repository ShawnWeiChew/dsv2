#ifndef OPS_H
#define OPS_H

#include "config.h"
#include "ds_state.h"
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

#define DS_4BIT_MATMUL_GROUPSIZE              64
#define DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32 8

// performs a softmax in place over the inner dimension
void softmax(float *in, size_t M, size_t N);

// rms norm over the inner dimension, broadcasts the element_wise_affine over the M dimension
void rms_norm(float *in, _Float16 *element_wise_affine, size_t M, size_t N, float eps);

void silu(float *in, size_t M, size_t N);

/**
 * On this version, the quantization format makes use of 4 bits for each weight, stored in uint32_t
 * this means that there are 32 / 4 = 8 floats in a 32-bit number.
 *
 * The group size is 64, which means that for every 8 uint32_ts, there is one scale and one offset
 *
 * This also supports a decode-style approach, where the matmul should be (seq_len, K)  @ (K, N)
 * but is instead done as (N, K) @ (K,) -> (N, 1), since the sequence length of decode is 1
 */
void ds_matmul_4bit(
    float *out, float *in, uint32_t *weights, _Float16 *scales, _Float16 *biases, size_t N, size_t K
);

/**
 * b (N, K) @ a (K,) -> c (N,)
 */
void matmul(float *c, float *a, _Float16 *b, size_t N, size_t K);

#define DS_YARN_ROTATION_FLOOR   1.0f
#define DS_YARN_ROTATION_CEILING 32.0f
#define DS_SCALE_FACTOR          40
#define DS_YARN_MSCALE           ((0.707 / 0.707))
#define DS_YARN_BASE             10000
#define DS_INTIAL_CONTEXT_LEN    4096.0f

typedef struct {
    int idx1;
    int idx2;
    bool idx_2_neg;
} YarnInterleavedAccessPattern;

typedef struct {
    // should be populated up till max sequence length
    float *cos;
    float *sin;
    // should span one hidden dim
    YarnInterleavedAccessPattern *access_pattern;
} YarnConstants;

/**
 * Generate sin and cos cache, both of which have dimensions of
 */
void setup_yarn_sin_cos_cache(
    DeepseekConfig *config, YarnConstants *yarn_constants, size_t cache_len
);

/**
 * M gives the outer position, N gives the position within the hidden dimension
 */
void yarn(float *in, float *out, size_t M, size_t N, YarnConstants *yarn_constants);

void free_yarn_sin_cos_cache(YarnConstants *yarn_constants);

void transpose(float *in, float *out, size_t M, size_t N);

typedef struct {
    float score;
    size_t idx;
} DSRoutedExpert;
void identify_topk(
    DSRoutedExpert top_experts[], float *gating_result, int num_experts_to_select,
    int num_experts_total
);

/**
 * Performs the MoE operations
 * - MoE gate
 * - Routing to top-k experts & Scaling of their outputs
 * - Routing through shared experts
 * - Summation
 */
void ds_moe_layer(
    DSRunningState *state, DeepseekConfig *config, DSMoELayerWeights *weights, float *in
);

/**
 * MLA operations
 * - kv_a_proj -> kv_down (cached) + k_rope (apply rope and cache)
 * - kv_b_proj
 * - q_proj ->
 */
void ds_mla_layer_naive(
    DSRunningState *state, DeepseekConfig *config, DSMLALayerWeights *weights, float *in,
    int layer_no, int seq_no, YarnConstants *yarn_constants
);

void ds_mlp_layer(
    DSRunningState *state, DeepseekConfig *config, DSMLPLayerWeights *weights, float *in
);

#endif