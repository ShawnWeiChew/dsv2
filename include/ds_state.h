/**
 * Handles all transformer related state, like weights, intermediate activations and KV caches
 */

#ifndef STATE_H
#define STATE_H

#include "config.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { DS_FILE_1 = 0, DS_FILE_2 = 1 } DSWeightsFileNo;

typedef struct {
    _Float16 *biases;
    _Float16 *scales;
    uint32_t *weights;
} DSEmbedLayerWeights;

typedef struct {
    _Float16 *input_layernorm;

    // down proj
    _Float16 *down_proj_biases;
    _Float16 *down_proj_scales;
    uint32_t *down_proj_weights;

    // swiglu
    _Float16 *gate_proj_biases;
    _Float16 *gate_proj_scales;
    uint32_t *gate_proj_weights;

    // up proj
    _Float16 *up_proj_biases;
    _Float16 *up_proj_scales;
    uint32_t *up_proj_weights;
} DSMLPLayerWeights;

typedef struct {
    _Float16 *post_attention_layernorm;
    _Float16 *kv_a_layernorm;

    // kva proj - compression
    _Float16 *kv_a_proj_biases;
    _Float16 *kv_a_proj_scales;
    uint32_t *kv_a_proj_weights;

    // kvb proj - uncompression
    _Float16 *kv_b_proj_biases;
    _Float16 *kv_b_proj_scales;
    uint32_t *kv_b_proj_weights;

    // o_proj
    _Float16 *o_proj_biases;
    _Float16 *o_proj_scales;
    uint32_t *o_proj_weights;

    // q_proj
    _Float16 *q_proj_biases;
    _Float16 *q_proj_scales;
    uint32_t *q_proj_weights;
} DSMLALayerWeights;

typedef struct {
    _Float16 *input_layernorm;

    // shared down
    _Float16 *shared_down_biases;
    _Float16 *shared_down_scales;
    uint32_t *shared_down_weights;

    // shared gate
    _Float16 *shared_gate_biases;
    _Float16 *shared_gate_scales;
    uint32_t *shared_gate_weights;

    // shared up
    _Float16 *shared_up_biases;
    _Float16 *shared_up_scales;
    uint32_t *shared_up_weights;

    _Float16 *moe_gate_weights;

    // routed down
    _Float16 *routed_down_biases;
    _Float16 *routed_down_scales;
    uint32_t *routed_down_weights;

    // routed gate
    _Float16 *routed_gate_biases;
    _Float16 *routed_gate_scales;
    uint32_t *routed_gate_weights;

    // routed up
    _Float16 *routed_up_biases;
    _Float16 *routed_up_scales;
    uint32_t *routed_up_weights;
} DSMoELayerWeights;

typedef struct {
    _Float16 *lm_head_biases;
    _Float16 *lm_head_scales;
    uint32_t *lm_head_weights;
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

    void *file1_base;
    size_t file1_size;
    void *file2_base;
    size_t file2_size;
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

void free_weights(DSWeights *weights);
#endif