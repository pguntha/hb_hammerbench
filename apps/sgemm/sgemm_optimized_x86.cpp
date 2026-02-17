// sgemm_optimized_x86.cpp
// Optimized SGEMM: C = A * B  (single-precision, MxK * KxN -> MxN)
//
// All tile parameters are configurable at compile time via -D flags:
//   -DTILE_M=16 -DTILE_K=64 -DTILE_N=64 -DMICRO_M=4
//
// Core pinning is configurable via -DPIN_CORE=N (default: no pinning)
//
// Build:  g++ -O3 -march=native -funroll-loops -DTILE_M=16 -DTILE_K=64
//           -DTILE_N=64 -DMICRO_M=4 -DPIN_CORE=1 -o sgemm_opt sgemm_optimized_x86.cpp -lm
// Run:    ./sgemm_opt [M] [K] [N]

#include "sgemm_common_x86.h"

// Tile size defaults
// Tuned for Xeon X5650 Westmere (CPU core on lab machine): 32 KB L1d, 64-byte cache lines.
//
// Default working-set budget (~24 KB, ~75% of L1d):
//   A tile: TILE_M × TILE_K × 4 bytes
//   B tile: TILE_K × TILE_N × 4 bytes
//   C tile: TILE_M × TILE_N × 4 bytes
//
// Override any/all at compile time with -DTILE_M=... etc.

#ifndef TILE_M
#define TILE_M 32
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef MICRO_M
#define MICRO_M 4
#endif

// Tiled, reordered, register-blocked SGEMM
//
// Key ideas:
//   - i-k-j order makes the innermost j-loop stride-1 on both B and C, which is ideal for cache lines and lets the compiler auto-vectorize to SSE (4 floats/cycle on Westmere, no AVX on X5650).
//   - Tiling keeps A/B/C blocks resident in L1 across the inner loops. Without tiling, each B element gets evicted and reloaded M times.
//   - Micro-tiling loads B[k][j] once and multiplies it with MICRO_M different A values, cutting memory bandwidth demand.

static void sgemm_optimized(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));

    for (int ii = 0; ii < M; ii += TILE_M) {
        int i_end = (ii + TILE_M < M) ? ii + TILE_M : M;

        for (int kk = 0; kk < K; kk += TILE_K) {
            int k_end = (kk + TILE_K < K) ? kk + TILE_K : K;

            for (int jj = 0; jj < N; jj += TILE_N) {
                int j_end = (jj + TILE_N < N) ? jj + TILE_N : N;

                // Micro-tiled inner computation: MICRO_M rows at a time
                int i = ii;
                for (; i + MICRO_M <= i_end; i += MICRO_M) {
                    for (int k = kk; k < k_end; k++) {
                        // Load MICRO_M values from column k of A
                        float a_vals[MICRO_M];
                        for (int m = 0; m < MICRO_M; m++) {
                            a_vals[m] = A[(i + m) * K + k];
                        }

                        // Broadcast each a_val across the j-strip
                        for (int j = jj; j < j_end; j++) {
                            float b_val = B[k * N + j];
                            for (int m = 0; m < MICRO_M; m++) {
                                C[(i + m) * N + j] += a_vals[m] * b_val;
                            }
                        }
                    }
                }

                // Leftover rows that don't fill a full micro-tile
                for (; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[i * K + k];
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }
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

    // Report configuration
    int tile_footprint = (TILE_M * TILE_K + TILE_K * TILE_N + TILE_M * TILE_N) * (int)sizeof(float);
    printf("=== OPTIMIZED SGEMM on x86 ===\n");
    printf("Matrix dimensions: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Tile sizes: TILE_M=%d, TILE_K=%d, TILE_N=%d, MICRO_M=%d\n",
           TILE_M, TILE_K, TILE_N, MICRO_M);
    printf("Tile footprint: %d bytes (%.1f KB / 32 KB L1d)\n",
           tile_footprint, tile_footprint / 1024.0);
    printf("Benchmark iterations: %d (reporting median)\n", NUM_ITERS);

    float *A, *B, *C;
    if (!alloc_and_init(&A, &B, &C, M, K, N)) return 1;

    // Warm up (populate caches, prime branch predictors)
    sgemm_optimized(A, B, C, M, K, N);

    double median = benchmark(sgemm_optimized, A, B, C, M, K, N);
    report_and_verify(median, A, B, C, M, K, N);

    free(A); free(B); free(C);
    return 0;
}
