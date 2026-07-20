#include "../include/ds_state.h"
#include "../thirdparty/json.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint64_t parse_json_for_offsets(struct json_value_s *root, const char *layer_name) {
    struct json_value_s *embed_layer_weights = json_object_get(root, layer_name);
    if (embed_layer_weights == NULL) {
        fprintf(stderr, "Could not find field: %s\n", layer_name);
        exit(1);
    }
    struct json_value_s *offset_numbers =
        json_object_get(embed_layer_weights, DS_WEIGHT_OFFSET_FIELD_NAME);
    struct json_array_s *offsets = json_value_as_array(offset_numbers);
    if (offsets != NULL) {

        struct json_number_s *val = json_value_as_number(offsets->start->value);
        char *endptr;
        uint64_t number = strtol(val->number, &endptr, 10);

        if (errno == ERANGE && (number == LONG_MAX || number == LONG_MIN)) {
            fprintf(stderr, "Overflow or underflow occurred for %s\n", layer_name);
            exit(1);
        } else if (endptr == val->number) {
            fprintf(stderr, "No digits were found for %s\n", layer_name);
            exit(1);
        } else if (*endptr != '\0') {
            fprintf(stderr, "Partial conversion for %s", layer_name);
            exit(1);
        }

        return number;
    } else {
        fprintf(stderr, "Could not read offsets for %s", layer_name);
        exit(1);
    }

    return -1;
}

const char *quantized_field_names[] = {"biases", "scales", "weight"};

__attribute__((format(printf, 4, 5))) static void load_single_field(
    void **weight, struct json_value_s *root, const char *data_base, const char *format, ...
) {
    char layer_name[128];
    va_list args;
    va_start(args, format);
    vsnprintf(layer_name, 128, format, args);
    va_end(args);

    uint64_t offset = parse_json_for_offsets(root, layer_name);
    *weight = (void *)(data_base + offset);
}

// Takes advantage of the fact that all the weights are just a pointer away
// and that they are always laid out in biases, scales, weights in the weight struct
__attribute__((format(printf, 4, 5))) static void load_all_quantized_fields(
    void **weights, struct json_value_s *root, const char *data_base, const char *format, ...
) {
    char layer_root_name[128];
    va_list args;
    va_start(args, format);
    vsnprintf(layer_root_name, 128, format, args);
    va_end(args);

    for (int i = 0; i < 3; i++) {
        char layer_name[256];
        snprintf(
            layer_name, sizeof(layer_name), "%s.%s", layer_root_name, quantized_field_names[i]
        );
        uint64_t offset = parse_json_for_offsets(root, layer_name);
        weights[i] = (void *)(data_base + offset);
    }
}

static void configure_offsets(DSWeights *weights) {
    // we are going to manually set this up with the assumption that there are 2 safetensor files
    int fd1 = open(DS_WEIGHTS_FILE_1, O_RDONLY);
    if (fd1 < 0) {
        perror("Could not open weights file 1");
        exit(1);
    }
    struct stat sb1;
    if (fstat(fd1, &sb1) < 0) {
        perror("Could not stat weights file 1");
        close(fd1);
        exit(1);
    }
    void *file1_base = mmap(NULL, sb1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
    if (file1_base == MAP_FAILED) {
        perror("Could not mmap weights file 1");
        close(fd1);
        exit(1);
    }
    close(fd1);

    weights->file1_base = file1_base;
    weights->file1_size = sb1.st_size;

    uint64_t header_size = *(uint64_t *)file1_base;
    printf("First header size is: %zu\n", header_size);

    struct json_value_s *root = json_parse((char *)file1_base + 8, header_size);
    if (root == NULL) {
        perror("Could not parse header JSON 1");
        exit(1);
    }

    // NOTE: the offsets in safetensors are given from offset after the header section, since there
    // are weights with offset 0
    const char *data_base_1 = (char *)file1_base + 8 + header_size;

    // the first file contains 15.5 layers of weights

    // the first contents are the embedding layer weights
    load_all_quantized_fields(
        (void **)&weights->embed.biases, root, data_base_1, "model.embed_tokens"
    );

    // the first layer is the dense layer, so there is no loop
    load_single_field(
        (void **)&weights->dense_layer.mlp.input_layernorm,
        root,
        data_base_1,
        "model.layers.0.input_layernorm.weight"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.mlp.down_proj_biases,
        root,
        data_base_1,
        "model.layers.0.mlp.down_proj"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.mlp.up_proj_biases,
        root,
        data_base_1,
        "model.layers.0.mlp.up_proj"
    );

    load_single_field(
        (void **)&weights->dense_layer.attn.kv_a_layernorm,
        root,
        data_base_1,
        "model.layers.0.self_attn.kv_a_layernorm.weight"
    );
    load_single_field(
        (void **)&weights->dense_layer.attn.post_attention_layernorm,
        root,
        data_base_1,
        "model.layers.0.post_attention_layernorm.weight"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.attn.kv_a_proj_biases,
        root,
        data_base_1,
        "model.layers.0.self_attn.kv_a_proj_with_mqa"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.attn.kv_b_proj_biases,
        root,
        data_base_1,
        "model.layers.0.self_attn.kv_b_proj"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.attn.o_proj_biases,
        root,
        data_base_1,
        "model.layers.0.self_attn.o_proj"
    );
    load_all_quantized_fields(
        (void **)&weights->dense_layer.attn.q_proj_biases,
        root,
        data_base_1,
        "model.layers.0.self_attn.q_proj"
    );

    // for the next 15 layers, load the weights
    for (int i = 0; i < 15; i++) {
        int layer_no = i + 1;

        load_single_field(
            (void **)&weights->moe_layers[i].moe.input_layernorm,
            root,
            data_base_1,
            "model.layers.%d.input_layernorm.weight",
            layer_no
        );

        load_single_field(
            (void **)&weights->moe_layers[i].moe.moe_gate_weights,
            root,
            data_base_1,
            "model.layers.%d.mlp.gate.weight",
            layer_no
        );

        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_down_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.shared_experts.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_gate_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.shared_experts.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_up_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.shared_experts.up_proj",
            layer_no
        );

        // .biases
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_down_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.switch_mlp.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_gate_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.switch_mlp.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_up_biases,
            root,
            data_base_1,
            "model.layers.%d.mlp.switch_mlp.up_proj",
            layer_no
        );

        load_single_field(
            (void **)&weights->moe_layers[i].attn.kv_a_layernorm,
            root,
            data_base_1,
            "model.layers.%d.self_attn.kv_a_layernorm.weight",
            layer_no
        );
        load_single_field(
            (void **)&weights->moe_layers[i].attn.post_attention_layernorm,
            root,
            data_base_1,
            "model.layers.%d.post_attention_layernorm.weight",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.kv_a_proj_biases,
            root,
            data_base_1,
            "model.layers.%d.self_attn.kv_a_proj_with_mqa",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.kv_b_proj_biases,
            root,
            data_base_1,
            "model.layers.%d.self_attn.kv_b_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.o_proj_biases,
            root,
            data_base_1,
            "model.layers.%d.self_attn.o_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.q_proj_biases,
            root,
            data_base_1,
            "model.layers.%d.self_attn.q_proj",
            layer_no
        );
    }

    // layer 16 (index 15) is a special layer split across files
    int layer_16_no = 16;
    int layer_16_idx = 15;

    // MLP switch_mlp (gate_proj and up_proj only)
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.routed_gate_biases,
        root,
        data_base_1,
        "model.layers.%d.mlp.switch_mlp.gate_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.routed_up_biases,
        root,
        data_base_1,
        "model.layers.%d.mlp.switch_mlp.up_proj",
        layer_16_no
    );

    // self_attn kv_a_layernorm
    load_single_field(
        (void **)&weights->moe_layers[layer_16_idx].attn.kv_a_layernorm,
        root,
        data_base_1,
        "model.layers.%d.self_attn.kv_a_layernorm.weight",
        layer_16_no
    );

    // self_attn projections
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].attn.kv_a_proj_biases,
        root,
        data_base_1,
        "model.layers.%d.self_attn.kv_a_proj_with_mqa",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].attn.kv_b_proj_biases,
        root,
        data_base_1,
        "model.layers.%d.self_attn.kv_b_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].attn.o_proj_biases,
        root,
        data_base_1,
        "model.layers.%d.self_attn.o_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].attn.q_proj_biases,
        root,
        data_base_1,
        "model.layers.%d.self_attn.q_proj",
        layer_16_no
    );

    free(root);

    // now we load the second safetensors file
    // we are going to manually set this up with the assumption that there are 2 safetensor files
    int fd2 = open(DS_WEIGHTS_FILE_2, O_RDONLY);
    if (fd2 < 0) {
        perror("Could not open weights file 2");
        exit(1);
    }
    struct stat sb2;
    if (fstat(fd2, &sb2) < 0) {
        perror("Could not stat weights file 2");
        close(fd2);
        exit(1);
    }
    void *file2_base = mmap(NULL, sb2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
    if (file2_base == MAP_FAILED) {
        perror("Could not mmap weights file 2");
        close(fd2);
        exit(1);
    }
    close(fd2);

    weights->file2_base = file2_base;
    weights->file2_size = sb2.st_size;

    uint64_t header_size2 = *(uint64_t *)file2_base;
    printf("Second header size is: %zu\n", header_size2);

    struct json_value_s *root2 = json_parse((char *)file2_base + 8, header_size2);
    if (root2 == NULL) {
        perror("Could not parse header JSON 2");
        exit(1);
    }

    const char *data_base_2 = (char *)file2_base + 8 + header_size2;

    // load the remaining fields into layer 16
    load_single_field(
        (void **)&weights->moe_layers[layer_16_idx].moe.input_layernorm,
        root2,
        data_base_2,
        "model.layers.%d.input_layernorm.weight",
        layer_16_no
    );

    load_single_field(
        (void **)&weights->moe_layers[layer_16_idx].moe.moe_gate_weights,
        root2,
        data_base_2,
        "model.layers.%d.mlp.gate.weight",
        layer_16_no
    );

    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.shared_down_biases,
        root2,
        data_base_2,
        "model.layers.%d.mlp.shared_experts.down_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.shared_gate_biases,
        root2,
        data_base_2,
        "model.layers.%d.mlp.shared_experts.gate_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.shared_up_biases,
        root2,
        data_base_2,
        "model.layers.%d.mlp.shared_experts.up_proj",
        layer_16_no
    );

    load_all_quantized_fields(
        (void **)&weights->moe_layers[layer_16_idx].moe.routed_down_biases,
        root2,
        data_base_2,
        "model.layers.%d.mlp.switch_mlp.down_proj",
        layer_16_no
    );

    load_single_field(
        (void **)&weights->moe_layers[layer_16_idx].attn.post_attention_layernorm,
        root2,
        data_base_2,
        "model.layers.%d.post_attention_layernorm.weight",
        layer_16_no
    );

    // load the rest of the layers (layers 17 to 26) from root2
    for (int i = 16; i < DS_N_LAYERS - 1; i++) {
        int layer_no = i + 1;

        load_single_field(
            (void **)&weights->moe_layers[i].moe.input_layernorm,
            root2,
            data_base_2,
            "model.layers.%d.input_layernorm.weight",
            layer_no
        );

        load_single_field(
            (void **)&weights->moe_layers[i].moe.moe_gate_weights,
            root2,
            data_base_2,
            "model.layers.%d.mlp.gate.weight",
            layer_no
        );

        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_down_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.shared_experts.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_gate_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.shared_experts.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.shared_up_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.shared_experts.up_proj",
            layer_no
        );

        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_down_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.switch_mlp.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_gate_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.switch_mlp.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].moe.routed_up_biases,
            root2,
            data_base_2,
            "model.layers.%d.mlp.switch_mlp.up_proj",
            layer_no
        );

        load_single_field(
            (void **)&weights->moe_layers[i].attn.kv_a_layernorm,
            root2,
            data_base_2,
            "model.layers.%d.self_attn.kv_a_layernorm.weight",
            layer_no
        );
        load_single_field(
            (void **)&weights->moe_layers[i].attn.post_attention_layernorm,
            root2,
            data_base_2,
            "model.layers.%d.post_attention_layernorm.weight",
            layer_no
        );

        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.kv_a_proj_biases,
            root2,
            data_base_2,
            "model.layers.%d.self_attn.kv_a_proj_with_mqa",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.kv_b_proj_biases,
            root2,
            data_base_2,
            "model.layers.%d.self_attn.kv_b_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.o_proj_biases,
            root2,
            data_base_2,
            "model.layers.%d.self_attn.o_proj",
            layer_no
        );
        load_all_quantized_fields(
            (void **)&weights->moe_layers[i].attn.q_proj_biases,
            root2,
            data_base_2,
            "model.layers.%d.self_attn.q_proj",
            layer_no
        );
    }

    load_single_field((void **)&weights->model_layernorm, root2, data_base_2, "model.norm.weight");

    // load the lm_head layers
    load_all_quantized_fields(
        (void **)&weights->lm_head.lm_head_biases, root2, data_base_2, "lm_head"
    );

    free(root2);
}

void load_weights(DSWeights *weights) {
    // first open up the header file to check the file size
    printf("Loading model weights...\n\n");

    configure_offsets(weights);
}

void free_weights(DSWeights *weights) {
    munmap(weights->file1_base, weights->file1_size);
    munmap(weights->file2_base, weights->file2_size);
}

// state management functions
void allocate_running_state(
    DSRunningState *state, DeepseekConfig *config, size_t max_sequence_len
) {
    state->max_sequence_len = max_sequence_len;

    // MLA layer
    state->kv_lora_rope_scratch =
        calloc(config->kv_lora_rank + config->qk_rope_head_dim, sizeof(float));
    state->k_rope_cache =
        calloc(config->n_layers * max_sequence_len * config->qk_rope_head_dim, sizeof(float));
    state->q_nope_rope_scratch =
        calloc(config->n_attn_heads * (config->qk_nope_head_dim + config->qk_rope_head_dim), sizeof(float));
    state->q_rope_scratch =
        calloc(config->n_attn_heads * config->qk_rope_head_dim, sizeof(float));
    state->kv_cache =
        calloc(config->n_layers * max_sequence_len * (config->hidden_dim * 2), sizeof(float));
    state->qk_attention_scores_scratch = calloc(max_sequence_len, sizeof(float));
    state->final_attention_score =
        calloc(config->n_attn_heads * config->qk_nope_head_dim, sizeof(float));
    state->mla_out_proj_res = calloc(config->hidden_dim, sizeof(float));

    // MoE layer
    state->topk_routing_results = calloc(config->n_routed_experts, sizeof(float));
    state->routed_expert_up_scratch = calloc(config->moe_hidden_size, sizeof(float));
    state->routed_expert_swiglu_scratch = calloc(config->moe_hidden_size, sizeof(float));
    state->routed_expert_down_scratch = calloc(config->hidden_dim, sizeof(float));

    state->shared_expert_up_scratch =
        calloc(config->moe_hidden_size * config->n_shared_experts, sizeof(float));
    state->shared_expert_swiglu_scratch =
        calloc(config->moe_hidden_size * config->n_shared_experts, sizeof(float));
    state->shared_expert_down_scratch = calloc(config->hidden_dim, sizeof(float));

    state->moe_ffn_sum = calloc(config->hidden_dim, sizeof(float));

    // MLP layer
    state->mlp_up_scratch = calloc(config->mlp_hidden, sizeof(float));
    state->mlp_swiglu_scratch = calloc(config->mlp_hidden, sizeof(float));
    state->mlp_down_scratch = calloc(config->hidden_dim, sizeof(float));
}

void free_running_state(DSRunningState *state) {
    // MLA layer
    free(state->kv_lora_rope_scratch);
    free(state->k_rope_cache);
    free(state->q_nope_rope_scratch);
    free(state->q_rope_scratch);
    free(state->kv_cache);
    free(state->qk_attention_scores_scratch);
    free(state->final_attention_score);
    free(state->mla_out_proj_res);

    // MoE layer
    free(state->topk_routing_results);
    free(state->routed_expert_up_scratch);
    free(state->routed_expert_swiglu_scratch);
    free(state->routed_expert_down_scratch);

    free(state->shared_expert_up_scratch);
    free(state->shared_expert_swiglu_scratch);
    free(state->shared_expert_down_scratch);

    free(state->moe_ffn_sum);

    // MLP layer
    free(state->mlp_up_scratch);
    free(state->mlp_swiglu_scratch);
    free(state->mlp_down_scratch);
}