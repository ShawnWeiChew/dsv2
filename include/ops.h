#ifndef OPS_H
#define OPS_H

#include "config.h"
#include <stdint.h>
#include <unistd.h>

#define DS_4BIT_MATMUL_GROUPSIZE              64
#define DS_4BIT_MATMUL_NUM_WEIGHTS_PER_UINT32 8

// performs a softmax in place over the inner dimension
void softmax(float *in, size_t M, size_t N);

// rms norm over the inner dimension, broadcasts the element_wise_affine over the M dimension
void rms_norm(float *in, _Float16 *element_wise_affine, size_t M, size_t N, float eps);

void silu(float *in, size_t M, size_t N);

/**
 * On this version, the quantization format makes use of 4 bits for each weight, stored in uint32_t
 * this means that there are 32 / 4 = 8 floats in a 32-bit number.
 *
 * The group size is 64, which means that for every 8 uint32_ts, there is one scale and one offset
 *
 * This also supports a decode-style approach, where the matmul should be (seq_len, K)  @ (K, N)
 * but is instead done as (N, K) @ (K,) -> (N, 1), since the sequence length of decode is 1
 */
void ds_matmul_4bit(
    float *out, float *in, uint32_t *weights, _Float16 *scales, _Float16 *biases, size_t N, size_t K
);

#endif