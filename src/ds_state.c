#include "../include/ds_state.h"
#include "../thirdparty/json.h"
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
// Takes advantage of the fact that all the weights are just a uint64_t away
// and that they are always laid out in biases, scales, weights in the weight struct
__attribute__((format(printf, 3, 4))) static void
load_all_quantized_fields(uint64_t *weights, struct json_value_s *root, const char *format, ...) {
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
        weights[i] = parse_json_for_offsets(root, layer_name);
    }
}

static void configure_offsets(DSWeights *weights) {
    // we are going to manually set this up with the assumption that there are 2 safetensor files
    FILE *file1 = fopen(DS_WEIGHTS_FILE_1, "rb");
    if (file1 == NULL) {
        perror("Could not open weights file");
        exit(1);
    }

    uint64_t header_size;
    fread(&header_size, sizeof(uint64_t), 1, file1);

    printf("Header size is: %zu\n", header_size);

    char *offset_mapping = (char *)malloc(header_size);
    if (offset_mapping == NULL) {
        perror("Could not allocate memory");
        exit(1);
    }
    fread(offset_mapping, sizeof(char), header_size, file1);

    // we dont need the file anymore for now
    fclose(file1);

    struct json_value_s *root = json_parse(offset_mapping, header_size);
    if (root == NULL) {
        perror("Could not parse header JSON");
        exit(1);
    }
    free(offset_mapping);

    // the first file contains 15.5 layers of weights

    // the first contents are the embedding layer weights
    load_all_quantized_fields(&weights->embed.biases_offset, root, "model.embed_tokens");

    // the first layer is the dense layer, so there is no loop
    weights->dense_layer.mlp.input_layernorm_offset =
        parse_json_for_offsets(root, "model.layers.0.input_layernorm.weight");
    load_all_quantized_fields(
        &weights->dense_layer.mlp.down_proj_biases_offset, root, "model.layers.0.mlp.down_proj"
    );
    load_all_quantized_fields(
        &weights->dense_layer.mlp.up_proj_biases_offset, root, "model.layers.0.mlp.up_proj"
    );

    weights->dense_layer.attn.kv_a_layernorm_offset =
        parse_json_for_offsets(root, "model.layers.0.self_attn.kv_a_layernorm.weight");
    weights->dense_layer.attn.post_attention_layernorm_offset =
        parse_json_for_offsets(root, "model.layers.0.post_attention_layernorm.weight");
    load_all_quantized_fields(
        &weights->dense_layer.attn.kv_a_proj_biases_offset,
        root,
        "model.layers.0.self_attn.kv_a_proj_with_mqa"
    );
    load_all_quantized_fields(
        &weights->dense_layer.attn.kv_b_proj_biases_offset,
        root,
        "model.layers.0.self_attn.kv_b_proj"
    );
    load_all_quantized_fields(
        &weights->dense_layer.attn.o_proj_biases_offset, root, "model.layers.0.self_attn.o_proj"
    );
    load_all_quantized_fields(
        &weights->dense_layer.attn.q_proj_biases_offset, root, "model.layers.0.self_attn.q_proj"
    );

    // for the next 15 layers, load the weights
    for (int i = 0; i < 15; i++) {
        int layer_no = i + 1;
        char buffer[128];
        snprintf(buffer, 128, "model.layers.%d.input_layernorm.weight", layer_no);
        weights->moe_layers[i].moe.input_layernorm_offset = parse_json_for_offsets(root, buffer);

        snprintf(buffer, 128, "model.layers.%d.mlp.gate.weight", layer_no);
        weights->moe_layers[i].moe.moe_gate_weights_offset = parse_json_for_offsets(root, buffer);

        load_all_quantized_fields(
            &weights->moe_layers[i].moe.shared_down_biases_offset,
            root,
            "model.layers.%d.mlp.shared_experts.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].moe.shared_gate_biases_offset,
            root,
            "model.layers.%d.mlp.shared_experts.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].moe.shared_up_biases_offset,
            root,
            "model.layers.%d.mlp.shared_experts.up_proj",
            layer_no
        );

        // .biases
        load_all_quantized_fields(
            &weights->moe_layers[i].moe.routed_down_biases_offset,
            root,
            "model.layers.%d.mlp.switch_mlp.down_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].moe.routed_gate_biases_offset,
            root,
            "model.layers.%d.mlp.switch_mlp.gate_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].moe.routed_up_biases_offset,
            root,
            "model.layers.%d.mlp.switch_mlp.up_proj",
            layer_no
        );

        snprintf(buffer, 128, "model.layers.%d.self_attn.kv_a_layernorm.weight", layer_no);
        weights->moe_layers[i].attn.kv_a_layernorm_offset = parse_json_for_offsets(root, buffer);
        snprintf(buffer, 128, "model.layers.%d.post_attention_layernorm.weight", layer_no);
        weights->moe_layers[i].attn.post_attention_layernorm_offset =
            parse_json_for_offsets(root, buffer);
        load_all_quantized_fields(
            &weights->moe_layers[i].attn.kv_a_proj_biases_offset,
            root,
            "model.layers.%d.self_attn.kv_a_proj_with_mqa",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].attn.kv_b_proj_biases_offset,
            root,
            "model.layers.%d.self_attn.kv_b_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].attn.o_proj_biases_offset,
            root,
            "model.layers.%d.self_attn.o_proj",
            layer_no
        );
        load_all_quantized_fields(
            &weights->moe_layers[i].attn.q_proj_biases_offset,
            root,
            "model.layers.%d.self_attn.q_proj",
            layer_no
        );
    }

    // layer 16 (index 15) is a special layer split across files
    int layer_16_no = 16;
    int layer_16_idx = 15;

    // MLP switch_mlp (gate_proj and up_proj only)
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].moe.routed_gate_biases_offset,
        root,
        "model.layers.%d.mlp.switch_mlp.gate_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].moe.routed_up_biases_offset,
        root,
        "model.layers.%d.mlp.switch_mlp.up_proj",
        layer_16_no
    );

    // self_attn kv_a_layernorm
    char buffer[128];
    snprintf(
        buffer, sizeof(buffer), "model.layers.%d.self_attn.kv_a_layernorm.weight", layer_16_no
    );
    weights->moe_layers[layer_16_idx].attn.kv_a_layernorm_offset =
        parse_json_for_offsets(root, buffer);

    // self_attn projections
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].attn.kv_a_proj_biases_offset,
        root,
        "model.layers.%d.self_attn.kv_a_proj_with_mqa",
        layer_16_no
    );
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].attn.kv_b_proj_biases_offset,
        root,
        "model.layers.%d.self_attn.kv_b_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].attn.o_proj_biases_offset,
        root,
        "model.layers.%d.self_attn.o_proj",
        layer_16_no
    );
    load_all_quantized_fields(
        &weights->moe_layers[layer_16_idx].attn.q_proj_biases_offset,
        root,
        "model.layers.%d.self_attn.q_proj",
        layer_16_no
    );

    free(root);

    // now we load the second safetensors file
}

void load_weights(DSWeights *weights) {
    // first open up the header file to check the file size
    printf("Loading model weights...\n\n");

    configure_offsets(weights);
}