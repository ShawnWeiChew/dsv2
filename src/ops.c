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
                sum += (expanded_weights[i] - bias) * scale * in[in_idx + i];
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