#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <polybench.h>
#include "gd_sk.h"

static void init_array(int n, DATA_TYPE POLYBENCH_2D(J,N,N,n,n), DATA_TYPE POLYBENCH_1D(sigma,N,n)) {
    srand(42); 
    for (int i = 0; i < n; i++) {
        sigma[i] = ((DATA_TYPE)rand() / RAND_MAX) * 2.0 - 1.0; 
        for (int j = 0; j < n; j++) {
            if (i == j) {
                J[i][j] = 0.0;
            } else {
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                J[i][j] = z / sqrt(n);
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            J[j][i] = J[i][j];
        }
    }
}

static void print_array(int n, DATA_TYPE POLYBENCH_1D(sigma,N,n)) {
    for (int i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, sigma[i]);
    }
}

static void kernel_gd(int n, int iters, DATA_TYPE POLYBENCH_2D(J,N,N,n,n), DATA_TYPE POLYBENCH_1D(sigma,N,n)) {
    DATA_TYPE lr = 0.1;

#pragma scop
    for (int t = 0; t < _PB_ITERS; t++) {
        for (int i = 0; i < _PB_N; i++) {
            DATA_TYPE gradient = 0.0;
            for (int j = 0; j < _PB_N; j++) {
                gradient -= J[i][j] * sigma[j];
            }
            
            DATA_TYPE next_val = sigma[i] - (lr * gradient);
            if (next_val > 1.0) next_val = 1.0;
            if (next_val < -1.0) next_val = -1.0;
            sigma[i] = next_val;
        }
    }
#pragma endscop
}

int main(int argc, char** argv) {
    int n = N;
    int iters = NUM_ITERATIONS;

    POLYBENCH_2D_ARRAY_DECL(J, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(sigma, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(J), POLYBENCH_ARRAY(sigma));

    polybench_start_instruments;
    kernel_gd(n, iters, POLYBENCH_ARRAY(J), POLYBENCH_ARRAY(sigma));
    polybench_stop_instruments;
    polybench_print_instruments;

    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(sigma)));

    POLYBENCH_FREE_ARRAY(J);
    POLYBENCH_FREE_ARRAY(sigma);

    return 0;
}