#ifndef OPS_H
#define OPS_H

#include <unistd.h>

// performs a softmax in place over the inner dimension
void softmax(float *in, size_t M, size_t N);

#endif