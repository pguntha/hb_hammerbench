// =============================================================================
// REVISION 1.1: Tile Size Tuning for HB Memory Hierarchy
// =============================================================================
// Changes from Rev 1.0:
//   - Reduced tile sizes to better match HB's vcache characteristics.
//   - TILE_N aligned to 16 (vcache line = 64 bytes = 16 floats) so that
//     each j-loop iteration processes exactly one or more full cache lines.
//   - Smaller overall working set to reduce vcache thrashing.
//
// The x86 tile sizes (32/64/64) were designed for a 32 KB private L1 cache.
// On HB, the vcache is shared across all cores and banks, and the access
// goes through a multi-hop network. We reduce tile sizes to keep the
// working set compact and aligned to the 64-byte cache line granularity.
// =============================================================================

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

// Multipod barrier state
volatile int done[NUM_POD_X]={0};
int alert = 0;

// Tuned tile sizes for HB:
//   A tile: TILE_M x TILE_K = 16 x 32 = 512 floats = 2 KB
//   B tile: TILE_K x TILE_N = 32 x 32 = 1024 floats = 4 KB
//   C tile: TILE_M x TILE_N = 16 x 32 = 512 floats = 2 KB
//   Total per-tile working set: ~8 KB
#define TILE_M 16
#define TILE_K 32
#define TILE_N 32
#define MICRO_M 4

extern "C"
int kernel(float *mat1, float *mat2, float *result, int pod_id)
{
    bsg_barrier_tile_group_init();
    bsg_barrier_tile_group_sync();

    bsg_barrier_multipod(pod_id, NUM_POD_X, done, &alert);
    bsg_cuda_print_stat_kernel_start();

    bsg_unroll(1)
    for (int iter = 0; iter < NITER; iter++) {
        float *A = &mat1[N*N*iter];
        float *B = &mat2[N*N*iter];
        float *C = &result[N*N*iter];

        // Zero out C
        for (int i = 0; i < N * N; i++) {
            C[i] = 0.0f;
        }

        // Tiled i-k-j loop with HB-tuned tile sizes.
        for (int ii = 0; ii < N; ii += TILE_M) {
            int i_end = (ii + TILE_M < N) ? ii + TILE_M : N;

            for (int kk = 0; kk < N; kk += TILE_K) {
                int k_end = (kk + TILE_K < N) ? kk + TILE_K : N;

                for (int jj = 0; jj < N; jj += TILE_N) {
                    int j_end = (jj + TILE_N < N) ? jj + TILE_N : N;

                    int i = ii;
                    for (; i + MICRO_M <= i_end; i += MICRO_M) {
                        for (int k = kk; k < k_end; k++) {
                            float a0 = A[(N*(i+0)) + k];
                            float a1 = A[(N*(i+1)) + k];
                            float a2 = A[(N*(i+2)) + k];
                            float a3 = A[(N*(i+3)) + k];

                            for (int j = jj; j < j_end; j++) {
                                float b_val = B[(N*k) + j];
                                C[(N*(i+0)) + j] += a0 * b_val;
                                C[(N*(i+1)) + j] += a1 * b_val;
                                C[(N*(i+2)) + j] += a2 * b_val;
                                C[(N*(i+3)) + j] += a3 * b_val;
                            }
                        }
                    }

                    // Remainder rows
                    for (; i < i_end; i++) {
                        for (int k = kk; k < k_end; k++) {
                            float a_val = A[(N*i) + k];
                            for (int j = jj; j < j_end; j++) {
                                C[(N*i) + j] += a_val * B[(N*k) + j];
                            }
                        }
                    }
                }
            }
        }
    }

    bsg_fence();
    bsg_barrier_tile_group_sync();
    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    bsg_barrier_tile_group_sync();
    return 0;
}

