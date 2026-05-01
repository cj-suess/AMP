#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <polybench.h>
#include "amp_sk.h"

// box-muller initialization for the SK landscape
static void init_array(int n, DATA_TYPE POLYBENCH_2D(J,N,N,n,n), DATA_TYPE POLYBENCH_1D(m,N,n), DATA_TYPE POLYBENCH_1D(m_old,N,n), DATA_TYPE POLYBENCH_1D(h,N,n)) {
    srand(42); // fixed seed for reproducible benchmarks
    for (int i = 0; i < n; i++) {
        m[i] = ((DATA_TYPE)rand() / RAND_MAX) * 0.002 - 0.001; // random start [-0.001, 0.001]
        m_old[i] = 0.0;
        h[i] = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) {
                J[i][j] = 0.0;
            } else {
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                J[i][j] = z / sqrt(n); // variance 1/N
            }
        }
    }
    // make j symmetric
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            J[j][i] = J[i][j];
        }
    }
}

// ensure polybench prints output to prevent DCE
static void print_array(int n, DATA_TYPE POLYBENCH_1D(m,N,n)) {
    for (int i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, m[i]);
    }
}

static void kernel_amp(int n, int iters, DATA_TYPE POLYBENCH_2D(J,N,N,n,n), DATA_TYPE POLYBENCH_1D(m,N,n), DATA_TYPE POLYBENCH_1D(m_old,N,n), DATA_TYPE POLYBENCH_1D(h,N,n)) {
    DATA_TYPE damping = 0.7;

#pragma scop
    for (int t = 0; t < _PB_ITERS; t++) {
        DATA_TYPE beta = 0.1 + (2.4 * (DATA_TYPE)t / (DATA_TYPE)(_PB_ITERS - 1));

        DATA_TYPE onsager_coef = 0.0;
        for (int i = 0; i < _PB_N; i++) {
            onsager_coef += (1.0 - (m[i] * m[i]));
        }
        onsager_coef /= (DATA_TYPE)_PB_N;

        for (int i = 0; i < _PB_N; i++) {
            DATA_TYPE j_dot_m = 0.0;
            for (int j = 0; j < _PB_N; j++) {
                j_dot_m += J[i][j] * m[j];
            }
            DATA_TYPE h_target = j_dot_m - (beta * onsager_coef * m_old[i]);
            h[i] = (damping * h[i]) + ((1.0 - damping) * h_target);
            m_old[i] = m[i];
        }

        for (int i = 0; i < _PB_N; i++) {
            m[i] = tanh(beta * h[i]);
        }
    }
#pragma endscop
}

// runs outside the polybench timer
static void greedy_quench(int n, DATA_TYPE POLYBENCH_2D(J,N,N,n,n), DATA_TYPE POLYBENCH_1D(m,N,n)) {
    int improved = 1;
    while (improved) {
        improved = 0;
        DATA_TYPE min_frust = 0.0;
        int worst_idx = -1;
        
        for (int i = 0; i < n; i++) {
            DATA_TYPE local_field = 0.0;
            for (int j = 0; j < n; j++) {
                local_field += J[i][j] * m[j];
            }
            DATA_TYPE frust = m[i] * local_field;
            if (frust < min_frust) {
                min_frust = frust;
                worst_idx = i;
            }
        }
        if (worst_idx != -1 && min_frust < 0.0) {
            m[worst_idx] *= -1.0;
            improved = 1;
        }
    }
}

int main(int argc, char** argv) {
    int n = N;
    int iters = NUM_ITERATIONS;

    POLYBENCH_2D_ARRAY_DECL(J, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(m, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(m_old, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(h, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(J), POLYBENCH_ARRAY(m), POLYBENCH_ARRAY(m_old), POLYBENCH_ARRAY(h));

    polybench_start_instruments;
    kernel_amp(n, iters, POLYBENCH_ARRAY(J), POLYBENCH_ARRAY(m), POLYBENCH_ARRAY(m_old), POLYBENCH_ARRAY(h));
    polybench_stop_instruments;
    polybench_print_instruments;

    for (int i = 0; i < n; i++) m[i] = (m[i] >= 0.0) ? 1.0 : -1.0;
    greedy_quench(n, POLYBENCH_ARRAY(J), POLYBENCH_ARRAY(m));

    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(m)));

    POLYBENCH_FREE_ARRAY(J);
    POLYBENCH_FREE_ARRAY(m);
    POLYBENCH_FREE_ARRAY(m_old);
    POLYBENCH_FREE_ARRAY(h);

    return 0;
}