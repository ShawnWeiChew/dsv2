#include "../include/ds_state.h"
#include "../include/ops.h"
#include "../include/tokenizer.h"
#include <assert.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static bool check_arrays(float *a, float *b, size_t length);

typedef struct {
    float *input;
    size_t input_len;

    float *output;

    float *expected_result;
    size_t expected_result_len;

    uint32_t *quantized_weights;
    size_t quantized_weights_len;

    _Float16 *scales;
    size_t scales_len;

    _Float16 *biases;
    size_t biases_len;
} QMatmulTest;
static void load_qmatmul_test_values(QMatmulTest *q);
static void free_qmatmul_test_values(QMatmulTest *q);

int main() {
    float a1[] = {
        -0.0518932007, -1.6193211079, -0.3191249073, -0.8132019639, 0.5285385847,  1.0677961111,
        -1.9641026258, -1.7329628468, 1.0941127539,  -2.4110176563, -1.1813437939, -1.1350206137,
        -0.4218506813, -0.5847893953, 1.2821590900,  -0.4851659238, 0.1028767899,  -0.9998973608,
        0.1337433457,  1.8961991072,  0.1004779562,  0.5042431951,  1.0238515139,  1.5704630613,
        -0.0759637728, 0.8442844748,  -0.6514309049, -2.3403193951, -0.3078316748, -0.1167789549,
    };

    float temp[sizeof(float) * 6 * 5];
    memcpy(temp, a1, sizeof(float) * 6 * 5);

    float expected_c1_softmax[] = {
        0.2365217209, 0.0493339375, 0.1810563505, 0.1104685888, 0.4226193130, 0.4615743756,
        0.0222589560, 0.0280470755, 0.4738827050, 0.0142367911, 0.0563496426, 0.0590213463,
        0.1204300448, 0.1023225710, 0.6618763804, 0.0622096136, 0.1120059788, 0.0371802673,
        0.1155171320, 0.6730870008, 0.0980138481, 0.1467710733, 0.2467763126, 0.4262789190,
        0.0821598768, 0.5091815591, 0.1141015962, 0.0210773852, 0.1608847827, 0.1947547793
    };
    softmax(temp, 6, 5);
    assert(check_arrays(temp, expected_c1_softmax, 6 * 5));

    float a2[] = {
        1.9269,  1.4873, 0.9007,  -2.1055, -0.7581, 1.0783, 0.8008,  1.6806,  0.3559, -0.6866,
        -0.4934, 0.2415, -0.2316, 0.0418,  -0.2516, 0.8599, -0.3097, -0.3957, 0.8034, -0.6216,
    };
    _Float16 ewa2[] = {-0.7656, -0.7505, 1.3525, 0.6865, -0.3276};
    rms_norm(a2, ewa2, 4, 5, DS_RMS_NORM_EPS);

    float expected_a2_rmsnorm[] = {
        -0.9626, -0.7283, 0.7949,  -0.9432, 0.1621, -0.8075, -0.5878, 2.2233,  0.2390, 0.2200,
        1.3026,  -0.6250, -1.0804, 0.0989,  0.2843, -1.0348, 0.3654,  -0.8413, 0.8670, 0.3201,
    };
    assert(check_arrays(a2, expected_a2_rmsnorm, 4 * 5));

    float a3[] = {0.8823, 0.9150, 0.3829, 0.9593, 0.3904};
    float a3_expected[] = {0.6240, 0.6533, 0.2276, 0.6936, 0.2329};
    silu(a3, 1, 5);
    assert(check_arrays(a3, a3_expected, 5));

    // NOTE: these matmul tests have to be removed because they were done with a different formula
    // for quantized matmul in mind QMatmulTest q; load_qmatmul_test_values(&q);
    // // this test in particular maps out the mlp down projection
    // ds_matmul_4bit(q.output, q.input, q.quantized_weights, q.scales, q.biases, 2048, 10944);

    // check_arrays(q.output, q.expected_result, 2048);

    // free_qmatmul_test_values(&q);
    puts("Passed ops checks");

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

    YarnConstants yc = {NULL, NULL, NULL};
    setup_yarn_sin_cos_cache(&deepseek_config, &yc, 128);

    float routing_scores[] = {
        0.3408,  0.2297,  1.7066,  1.3925,  0.0561,  0.2836, -0.5602, 1.4158,  1.6667, -0.2644,
        -1.1022, -0.8187, -1.0842, -0.0052, -1.6498, 0.9515, 0.7178,  -2.0773, 1.6672, 0.5206,
    };
    DSRoutedExpert routed_experts[6];
    identify_topk(routed_experts, routing_scores, 6, 20);

    DSRoutedExpert expected_routed_results[] = {
        {.score = 1.7066210508346558, .idx = 2},
        {.score = 1.6671867370605469, .idx = 18},
        {.score = 1.666737675666809, .idx = 8},
        {.score = 1.4157644510269165, .idx = 7},
        {.score = 1.3925154209136963, .idx = 3},
        {.score = 0.9515208005905151, .idx = 15},
    };

    for (int i = 0; i < 6; i++) {
        assert(routed_experts[i].idx == expected_routed_results[i].idx);
        float diff = routed_experts[i].score - expected_routed_results[i].score;
        assert((diff < 0 ? -diff : diff) < 1E-3);
    }

    puts("Passed routing checks");

    // now we are moving on to the model serving checks
    DSWeights weights;
    load_weights(&weights);

#include "test_moe_data.h"

    DSRunningState state;
    allocate_running_state(&state, &deepseek_config, 2048);

    ds_moe_layer(&state, &deepseek_config, &weights.moe_layers[0].moe, x);

    assert(check_arrays(state.moe_ffn_sum, moe_ffn_expected_result, 2048));

    ds_mla_layer_naive(&state, &deepseek_config, &weights.dense_layer.attn, mla_input, 0, 0, &yc);
    assert(check_arrays(state.mla_out_proj_res, mla_expected_output, 2048));

    ds_get_quantized_embeddings(&state, &deepseek_config, &weights.embed, DS_TOKENIZER_BOS);
    assert(check_arrays(state.token_embedding_scratch, embed_bos_out, 2048));

    free_yarn_sin_cos_cache(&yc);
    free_running_state(&state);
    free_weights(&weights);

    puts("Passed layer unit tests");
}

static bool check_arrays(float *a, float *b, size_t length) {
    for (int i = 0; i < length; i++) {
        float diff = a[i] - b[i];
        // NOTE: I have to give more allowance here becuase internally _Float16 is done by casting
        // to FP32, then doing the ops However, to my understanding, MLX does ops natively in FP16
        // do there will be some divergence in output
        if ((diff < 0 ? -diff : diff) >= 1E-2) {
            fprintf(stderr, "\n====================\n");
            fprintf(stderr, "Element %d was not equal. Expected %f but got %f", i, b[i], a[i]);
            fprintf(stderr, "\n====================\n");
            return false;
        }
    }

    return true;
}

static void *open_file_and_mmap(char *filename, size_t *len) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Could not open %s", filename);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *file_in_mmap = mmap(NULL, *len, PROT_READ, MAP_SHARED, fileno(f), 0);
    if (file_in_mmap == NULL) {
        perror("Could not mmap file");
        exit(1);
    }

    return file_in_mmap;
}

static void load_qmatmul_test_values(QMatmulTest *q) {
    q->input = open_file_and_mmap("layer0_input.bin", &q->input_len);
    q->biases = open_file_and_mmap("layer0_bias.bin", &q->biases_len);
    q->quantized_weights = open_file_and_mmap("layer0_weight.bin", &q->quantized_weights_len);
    // printf("%lu\n", q->quantized_weights_len);
    q->scales = open_file_and_mmap("layer0_scales.bin", &q->scales_len);
    q->expected_result = open_file_and_mmap("layer0_output.bin", &q->expected_result_len);

    q->output = calloc(2048, sizeof(float));
    if (q->output == NULL) {
        perror("Could not allocate output space");
        exit(1);
    }
}

static void free_qmatmul_test_values(QMatmulTest *q) {
    munmap(q->input, q->input_len);
    munmap(q->biases, q->biases_len);
    munmap(q->quantized_weights, q->quantized_weights_len);
    munmap(q->scales, q->scales_len);
    munmap(q->expected_result, q->expected_result_len);
    free(q->output);
}
