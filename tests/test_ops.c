#include "../include/ops.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

bool check_arrays(float *a, float *b, size_t length) {
    for (int i = 0; i < length; i++) {
        float diff = a[i] - b[i];
        if ((diff < 0 ? -diff : diff) >= 1E-3) {
            fprintf(stderr, "\n====================\n");
            fprintf(stderr, "Element %d was not equal. Expected %f but got %f", i, b[i], a[i]);
            fprintf(stderr, "\n====================\n");
            return false;
        }
    }

    return true;
}

static _Float16 mul(_Float16 a, _Float16 b) { return a * b; }

static float mul_mixed(_Float16 a, float b) { return a * b; }

void verify_floating_point_ops() { assert(mul(3.5145, 4.1293) == mul(3.5145, 4.1293)); }

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

    puts("Passed ops checks");

    verify_floating_point_ops();
}