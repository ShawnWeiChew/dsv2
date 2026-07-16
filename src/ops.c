#include "../include/ops.h"
#include <math.h>

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