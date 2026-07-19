#include "../include/ds_state.h"
#include "../include/ops.h"
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

    float a4[] = {
        1.9269e+00,  1.4873e+00,  9.0072e-01,  -2.1055e+00, 6.7842e-01,  -1.2345e+00, -4.3067e-02,
        -1.6047e+00, -7.5214e-01, 1.6487e+00,  -3.9248e-01, -1.4036e+00, -7.2788e-01, -5.5943e-01,
        -7.6884e-01, 7.6245e-01,  1.6423e+00,  -1.5960e-01, -4.9740e-01, 4.3959e-01,  -7.5813e-01,
        1.0783e+00,  8.0080e-01,  1.6806e+00,  1.2791e+00,  1.2964e+00,  6.1047e-01,  1.3347e+00,
        -2.3162e-01, 4.1759e-02,  -2.5158e-01, 8.5986e-01,  -1.3847e+00, -8.7124e-01, -2.2337e-01,
        1.7174e+00,  3.1888e-01,  -4.2452e-01, 3.0572e-01,  -7.7459e-01, -1.5576e+00, 9.9564e-01,
        -8.7979e-01, -6.0114e-01, -1.2742e+00, 2.1228e+00,  -1.2347e+00, -4.8791e-01, -9.1382e-01,
        -6.5814e-01, 7.8024e-02,  5.2581e-01,  -4.8799e-01, 1.1914e+00,  -8.1401e-01, -7.3599e-01,
        -1.4032e+00, 3.6004e-02,  -6.3477e-02, 6.7561e-01,  -9.7807e-02, 1.8446e+00,  -1.1845e+00,
        1.3835e+00,  1.4451e+00,  8.5641e-01,  2.2181e+00,  5.2317e-01,  3.4665e-01,  -1.9733e-01,
        -1.0546e+00, 1.2780e+00,  -1.7219e-01, 5.2379e-01,  5.6622e-02,  4.2630e-01,  5.7501e-01,
        -6.4172e-01, -2.2064e+00, -7.5080e-01, 1.0868e-02,  -3.3874e-01, -1.3407e+00, -5.8537e-01,
        5.3619e-01,  5.2462e-01,  1.1412e+00,  5.1644e-02,  7.4395e-01,  -4.8158e-01, -1.0495e+00,
        6.0390e-01,  -1.7223e+00, -8.2777e-01, 1.3347e+00,  4.8354e-01,  -2.5095e+00, 4.8800e-01,
        7.8459e-01,  2.8647e-02,  6.4076e-01,  5.8325e-01,  1.0669e+00,  -4.5015e-01, -1.8527e-01,
        7.5276e-01,  4.0476e-01,  1.7847e-01,  2.6491e-01,  1.2732e+00,  -1.3109e-03, -3.0360e-01,
        -1.4570e+00, -1.0234e-01, -5.9915e-01, 4.7706e-01,  7.2618e-01,  9.1152e-02,  -3.8907e-01,
        5.2792e-01,  -1.2685e-02, 2.4084e-01,  1.3254e-01,  7.6424e-01,  1.0950e+00,  3.3989e-01,
        7.1997e-01,  4.1141e-01,  1.9312e+00,  1.0119e+00,  -1.4364e+00, -1.1299e+00, -1.3603e-01,
        1.6354e+00,  6.5474e-01,  5.7600e-01,  1.1415e+00,  1.8565e-02,  -1.8058e+00, 9.2543e-01,
        -3.7534e-01, 1.0331e+00,  -6.8665e-01, 6.3681e-01,  -9.7267e-01, 9.5846e-01,  1.6192e+00,
        1.4506e+00,  2.6948e-01,  -2.1038e-01, -7.3280e-01, 1.0430e-01,  3.4875e-01,  9.6759e-01,
        -4.6569e-01, 1.6048e+00,  -2.4801e+00, -4.1754e-01, -1.1955e+00, 8.1234e-01,  -1.9006e+00,
        2.2858e-01,  2.4859e-02,  -3.4595e-01, 2.8683e-01,  -7.3084e-01, 1.7482e-01,  -1.0939e+00,
        -1.6022e+00, 1.3529e+00,  1.2888e+00,  5.2295e-02,  -1.5469e+00, 7.5671e-01,  7.7552e-01,
        2.0265e+00,  3.5818e-02,  1.2059e-01,  -8.0566e-01, -2.0758e-01, -9.3195e-01, -1.5910e+00,
        -1.1360e+00, -5.2260e-01, -5.1877e-01, -1.5013e+00, -1.9267e+00, 1.2785e-01,  1.0229e+00,
        -5.5580e-01, 7.0427e-01,  7.0988e-01,
    };

    float a4_expected[] = {
        1.9269e+00,  9.0072e-01,  6.7842e-01,  -4.3067e-02, -7.5214e-01, -3.9248e-01, -7.2788e-01,
        -7.6884e-01, 1.6423e+00,  -4.9740e-01, -7.5813e-01, 8.0080e-01,  1.2791e+00,  6.1047e-01,
        -2.3162e-01, -2.5158e-01, -1.3847e+00, -2.2337e-01, 3.1888e-01,  3.0572e-01,  -1.5576e+00,
        -8.7979e-01, -1.2742e+00, -1.2347e+00, -9.1382e-01, 7.8024e-02,  -4.8799e-01, -8.1401e-01,
        -1.4032e+00, -6.3477e-02, -9.7807e-02, -1.1845e+00, 1.4873e+00,  -2.1055e+00, -1.2345e+00,
        -1.6047e+00, 1.6487e+00,  -1.4036e+00, -5.5943e-01, 7.6245e-01,  -1.5960e-01, 4.3959e-01,
        1.0783e+00,  1.6806e+00,  1.2964e+00,  1.3347e+00,  4.1759e-02,  8.5986e-01,  -8.7124e-01,
        1.7174e+00,  -4.2452e-01, -7.7459e-01, 9.9564e-01,  -6.0114e-01, 2.1228e+00,  -4.8791e-01,
        -6.5814e-01, 5.2581e-01,  1.1914e+00,  -7.3599e-01, 3.6004e-02,  6.7561e-01,  1.8446e+00,
        1.3835e+00,  6.0163e-02,  1.2665e+00,  3.9848e-01,  -1.4853e+00, -3.2654e-01, -4.5109e-02,
        6.7945e-01,  -2.0870e+00, 4.4632e-02,  -1.2931e+00, 5.0585e-01,  1.1383e+00,  7.5663e-01,
        -1.0604e+00, -1.7119e+00, 1.3306e+00,  -2.5122e+00, 7.8448e-01,  6.3944e-01,  1.0675e+00,
        -1.8586e-01, 4.0468e-01,  2.6468e-01,  -1.3007e-03, -1.4570e+00, -5.9916e-01, 7.2618e-01,
        -3.8907e-01, -1.2687e-02, 1.3253e-01,  1.0950e+00,  7.1997e-01,  1.6788e+00,  1.8946e+00,
        1.7877e-02,  7.3439e-01,  4.4427e-01,  4.2767e-01,  -5.2989e-01, -1.0375e+00, -3.3597e-01,
        -6.8417e-01, 5.5393e-01,  9.6108e-02,  -4.6142e-01, 5.8451e-01,  -8.4914e-01, 4.9465e-01,
        4.7419e-01,  3.1442e-02,  5.8469e-01,  -4.4869e-01, 7.5261e-01,  1.7863e-01,  1.2732e+00,
        -3.0360e-01, -1.0237e-01, 4.7705e-01,  9.1162e-02,  5.2791e-01,  2.4084e-01,  7.6424e-01,
        3.3990e-01,  4.1141e-01,  -1.7237e+00, 1.0251e+00,  -1.5340e+00, 5.1385e-03,  9.0974e-01,
        -2.0291e+00, -7.1158e-01, -8.3021e-01, -1.1437e+00, 1.3843e+00,  2.9139e-01,  -7.3870e-01,
        2.9626e-01,  -5.2435e-01, -2.4690e+00, -1.2088e+00, -1.9030e+00, 2.7323e-02,  2.9012e-01,
        1.7782e-01,  -1.6043e+00, 1.2888e+00,  -1.5471e+00, 7.7538e-01,  3.5812e-02,  -8.0566e-01,
        -9.3190e-01, -1.1360e+00, -5.1875e-01, -1.9267e+00, 1.0229e+00,  7.0427e-01,  1.3349e+00,
        -1.5129e+00, 5.8290e-01,  8.7203e-01,  6.8975e-01,  -1.4099e-03, 8.3774e-01,  4.3333e-01,
        7.4611e-01,  1.6763e+00,  -1.7880e-01, 4.6870e-02,  9.8494e-01,  1.5866e+00,  -4.7915e-01,
        7.9230e-01,  2.0766e-01,  -3.4576e-01, -7.2954e-01, -1.0934e+00, 1.3504e+00,  5.3365e-02,
        7.5616e-01,  2.0266e+00,  1.2059e-01,  -2.0761e-01, -1.5910e+00, -5.2262e-01, -1.5013e+00,
        1.2783e-01,  -5.5579e-01, 7.0988e-01,
    };

    float a4_out[3 * 64];
    yarn(a4, a4_out, 3, 64, &yc);
    assert(check_arrays(a4_out, a4_expected, 3 * 64));
    free_yarn_sin_cos_cache(&yc);
    puts("Passed YaRN checks");

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
    allocate_running_state(&state, &deepseek_config);

    ds_moe_layer(&state, &deepseek_config, &weights.moe_layers[0].moe, x);

    assert(check_arrays(state.moe_ffn_sum, moe_ffn_expected_result, 2048));

    free_running_state(&state);
    free_weights(&weights);
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
