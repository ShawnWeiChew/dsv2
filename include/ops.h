#ifndef OPS_H
#define OPS_H

#include "config.h"
#include <unistd.h>

// performs a softmax in place over the inner dimension
void softmax(float *in, size_t M, size_t N);

// rms norm over the inner dimension, broadcasts the element_wise_affine over the M dimension
void rms_norm(float *in, _Float16 *element_wise_affine, size_t M, size_t N, float eps);

void silu(float *in, size_t M, size_t N);

#endif