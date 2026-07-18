/**
 * Handles all transformer related state, like weights, intermediate activations and KV caches
 */

#ifndef STATE_H
#define STATE_H

#include "config.h"
#include <stdint.h>

typedef struct {
    uint64_t biases_offset;
    uint64_t scales_offset;
    uint64_t weights_offset;
} DSEmbedLayerWeights;

typedef struct {
    uint64_t input_layernorm_offset;

    // down proj
    uint64_t down_proj_biases_offset;
    uint64_t down_proj_scales_offset;
    uint64_t down_proj_weights_offset;

    // swiglu
    uint64_t gate_proj_biases_offset;
    uint64_t gate_proj_scales_offset;
    uint64_t gate_proj_weights_offset;

    // up proj
    uint64_t up_proj_biases_offset;
    uint64_t up_proj_scales_offset;
    uint64_t up_proj_weights_offset;
} DSMLPLayerWeights;

typedef struct {
    uint64_t post_attention_layernorm_offset;
    uint64_t kv_a_layernorm_offset;

    // kva proj - compression
    uint64_t kv_a_proj_biases_offset;
    uint64_t kv_a_proj_scales_offset;
    uint64_t kv_a_proj_weights_offset;

    // kvb proj - uncompression
    uint64_t kv_b_proj_biases_offset;
    uint64_t kv_b_proj_scales_offset;
    uint64_t kv_b_proj_weights_offset;

    // o_proj
    uint64_t o_proj_biases_offset;
    uint64_t o_proj_scales_offset;
    uint64_t o_proj_weights_offset;

    // q_proj
    uint64_t q_proj_biases_offset;
    uint64_t q_proj_scales_offset;
    uint64_t q_proj_weights_offset;
} DSMLALayerWeights;

typedef struct {
    uint64_t input_layernorm_offset;

    // shared down
    uint64_t shared_down_biases_offset;
    uint64_t shared_down_scales_offset;
    uint64_t shared_down_weights_offset;

    // shared gate
    uint64_t shared_gate_biases_offset;
    uint64_t shared_gate_scales_offset;
    uint64_t shared_gate_weights_offset;

    // shared up
    uint64_t shared_up_biases_offset;
    uint64_t shared_up_scales_offset;
    uint64_t shared_up_weights_offset;

    uint64_t moe_gate_weights_offset;

    // routed down
    uint64_t routed_down_biases_offset;
    uint64_t routed_down_scales_offset;
    uint64_t routed_down_weights_offset;

    // routed gate
    uint64_t routed_gate_biases_offset;
    uint64_t routed_gate_scales_offset;
    uint64_t routed_gate_weights_offset;

    // routed up
    uint64_t routed_up_biases_offset;
    uint64_t routed_up_scales_offset;
    uint64_t routed_up_weights_offset;
} DSMoELayerWeights;

typedef struct {
    uint64_t lm_head_biases_offset;
    uint64_t lm_head_scales_offset;
    uint64_t lm_head_weights_offset;
} DSLmHeadWeights;

typedef struct {
    DSMLPLayerWeights mlp;
    DSMLALayerWeights attn;
} DSDenseLayerWeights;

// NOTE: layer 16 has a slightly different format, cos the safetensors file is cut in half
typedef struct {
    DSMoELayerWeights moe;
    DSMLALayerWeights attn;
} DSExpertLayerWeights;

typedef struct {
    DSEmbedLayerWeights embed;

    DSDenseLayerWeights dense_layer;

    DSExpertLayerWeights moe_layers[DS_N_LAYERS - 1];

    DSLmHeadWeights lm_head;
} DSWeights;

#define DS_WEIGHTS_FILE_1           "model-00001-of-00002.safetensors"
#define DS_WEIGHTS_FILE_2           "model-00002-of-00002.safetensors"

#define DS_WEIGHT_OFFSET_FIELD_NAME "data_offsets"

/**
 * The safetensors format is:
 * 8 bytes for size of header, N
 *
 * N bytes for header which tells you the offset for all the other weights
 *
 * Based on inspection of the file, it seems like the weights are scattered all about, so we have to
 * resort to tracking the offsets. Note sure if this is an MLX problem or an issue with all
 * safetensors
 */
void load_weights(DSWeights *weights);

#endif