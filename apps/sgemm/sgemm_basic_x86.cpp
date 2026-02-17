// sgemm_basic_x86.cpp
// Naive SGEMM: C = A * B  (single-precision, MxK * KxN -> MxN)
//
// Core pinning is configurable via -DPIN_CORE=N (default: no pinning)
//
// Build:  g++ -O2 -DPIN_CORE=1 -o sgemm_basic sgemm_basic_x86.cpp -lm
// Run:    ./sgemm_basic [M] [K] [N]

#include "sgemm_common_x86.h"

// Naive triple-nested loop (i-j-k order)
//
// Why this is slow:
//   - B is accessed column-wise (stride N) in the innermost k-loop - terrible spatial locality, every k-step lands on a different cache line.
//   - No tiling: working set spans entire rows/columns, blows past L1 cache.
//   - No register reuse: C[i][j] is read/written every k iteration.
//   - No unrolling: one FMA's worth of work per loop iteration, poor compute-to-overhead ratio.
//   - Pure scalar: ignores SSE/SSE4.2 SIMD (wastes 75% of FP throughput).

static void sgemm_basic(const float* A, const float* B, float* C,
                         int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                // A[i][k]: stride-1 as k increments (good)
                // B[k][j]: stride-N as k increments (bad)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    // Core pinning
#ifdef PIN_CORE
    pin_to_core(PIN_CORE);
#endif

    int M, K, N;
    parse_dims(argc, argv, M, K, N);

    printf("=== BASIC (Naive) SGEMM on x86 ===\n");
    printf("Matrix dimensions: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Benchmark iterations: %d (reporting median)\n", NUM_ITERS);

    float *A, *B, *C;
    if (!alloc_and_init(&A, &B, &C, M, K, N)) return 1;

    // Warm up (populate caches, branch predictors, etc.)
    sgemm_basic(A, B, C, M, K, N);

    double median = benchmark(sgemm_basic, A, B, C, M, K, N);
    report_and_verify(median, A, B, C, M, K, N);

    free(A); free(B); free(C);
    return 0;
}
