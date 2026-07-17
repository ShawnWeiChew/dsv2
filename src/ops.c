#include "../include/ops.h"
#include <math.h>
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