#ifndef _GD_SK_H
#define _GD_SK_H
#include "polybench.h"

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define LARGE_DATASET
#endif

#ifdef LARGE_DATASET
#define N 1000
#define NUM_ITERATIONS 1000
#endif

#define _PB_N POLYBENCH_LOOP_BOUND(N,n)
#define _PB_ITERS POLYBENCH_LOOP_BOUND(NUM_ITERATIONS,num_iterations)

#ifndef DATA_TYPE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

#endif /* !_GD_SK_H */