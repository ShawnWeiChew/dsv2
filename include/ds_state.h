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
    _Float16 *biases;  // (102400, 32)
    _Float16 *scales;  // (102400, 32)
    uint32_t *weights; // (102400, 256)
} DSEmbedLayerWeights;

typedef struct {
    _Float16 *input_layernorm; // (2048)

    // down proj
    _Float16 *down_proj_biases;  // (2048, 171)
    _Float16 *down_proj_scales;  // (2048, 171)
    uint32_t *down_proj_weights; // (2048, 1368)

    // swiglu
    _Float16 *gate_proj_biases;  // (10944, 32)
    _Float16 *gate_proj_scales;  // (10944, 32)
    uint32_t *gate_proj_weights; // (10944, 256)

    // up proj
    _Float16 *up_proj_biases;  // (10944, 32)
    _Float16 *up_proj_scales;  // (10944, 32)
    uint32_t *up_proj_weights; // (10944, 256)
} DSMLPLayerWeights;

typedef struct {
    _Float16 *post_attention_layernorm; // (2048)
    _Float16 *kv_a_layernorm;           // (512)

    // kva proj - compression
    _Float16 *kv_a_proj_biases;  // (576, 32)
    _Float16 *kv_a_proj_scales;  // (576, 32)
    uint32_t *kv_a_proj_weights; // (576, 256)

    // kvb proj - uncompression
    _Float16 *kv_b_proj_biases;  // (4096, 8)
    _Float16 *kv_b_proj_scales;  // (4096, 8)
    uint32_t *kv_b_proj_weights; // (4096, 64)

    // o_proj
    _Float16 *o_proj_biases;  // (2048, 32)
    _Float16 *o_proj_scales;  // (2048, 32)
    uint32_t *o_proj_weights; // (2048, 256)

    // q_proj
    _Float16 *q_proj_biases;  // (3072, 32)
    _Float16 *q_proj_scales;  // (3072, 32)
    uint32_t *q_proj_weights; // (3072, 256)
} DSMLALayerWeights;

typedef struct {
    _Float16 *input_layernorm; // (2048)

    // shared down
    _Float16 *shared_down_biases;  // (2048, 44)
    _Float16 *shared_down_scales;  // (2048, 44)
    uint32_t *shared_down_weights; // (2048, 352)

    // shared gate
    _Float16 *shared_gate_biases;  // (2816, 32)
    _Float16 *shared_gate_scales;  // (2816, 32)
    uint32_t *shared_gate_weights; // (2816, 256)

    // shared up
    _Float16 *shared_up_biases;  // (2816, 32)
    _Float16 *shared_up_scales;  // (2816, 32)
    uint32_t *shared_up_weights; // (2816, 256)

    _Float16 *moe_gate_weights; // (64, 2048)

    // routed down
    _Float16 *routed_down_biases;  // (64, 2048, 22)
    _Float16 *routed_down_scales;  // (64, 2048, 22)
    uint32_t *routed_down_weights; // (64, 2048, 176)

    // routed gate
    _Float16 *routed_gate_biases;  // (64, 1408, 32)
    _Float16 *routed_gate_scales;  // (64, 1408, 32)
    uint32_t *routed_gate_weights; // (64, 1408, 256)

    // routed up
    _Float16 *routed_up_biases;  // (64, 1408, 32)
    _Float16 *routed_up_scales;  // (64, 1408, 32)
    uint32_t *routed_up_weights; // (64, 1408, 256)
} DSMoELayerWeights;

typedef struct {
    _Float16 *lm_head_biases;  // (102400, 32)
    _Float16 *lm_head_scales;  // (102400, 32)
    uint32_t *lm_head_weights; // (102400, 256)
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

    _Float16 *model_layernorm; // (2048)
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

typedef struct {
    // MLA layer
    float *kv_lora_cache; // (layer, sequence_number, 512), q here stands for quantized
    float *k_rope_cache;  // (layer, sequence_number, 64)

    // MoE layer
    float *topk_routing_results;         // (64,)
    float *routed_expert_up_scratch;     // (1408,)
    float *routed_expert_swiglu_scratch; // (1408,)
    float *routed_expert_down_scratch;   // (2048,)

    float *shared_expert_up_scratch;     // (1408 * 2,)
    float *shared_expert_swiglu_scratch; // (1408 * 2,)
    float *shared_expert_down_scratch;   // (2048,)

    float *moe_ffn_sum; // (2048,)
} DSRunningState;

/**
 * Use max_sequence_len argument to prevent allocating a huge context window that you dont really
 * use
 */
void allocate_running_state(DSRunningState *state, DeepseekConfig *config, size_t max_sequence_len);
void free_running_state(DSRunningState *state);

#endif