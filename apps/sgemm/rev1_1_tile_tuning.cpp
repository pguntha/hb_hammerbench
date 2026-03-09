// Rev 1.1: Serial SGEMM — reduced tile sizes (16/32/32) tuned for HB vcache.
// Only tile 0 computes; all others idle at barriers.

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X]={0};
int alert = 0;

#define TILE_M 16
#define TILE_K 32
#define TILE_N 32
#define MICRO_M 4

extern "C"
int kernel(float *mat1, float *mat2, float *result, int pod_id)
{
    bsg_barrier_tile_group_init();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 1: barrier_init done\n");

    bsg_barrier_multipod(pod_id, NUM_POD_X, done, &alert);

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 2: multipod barrier done\n");

    bsg_cuda_print_stat_kernel_start();

    if (__bsg_id == 0) {
        bsg_printf("[DBG] checkpoint 3: N=%d NITER=%d TILE_M=%d TILE_K=%d TILE_N=%d MICRO_M=%d\n",
                   N, NITER, TILE_M, TILE_K, TILE_N, MICRO_M);

        bsg_unroll(1)
        for (int iter = 0; iter < NITER; iter++) {
            float *A = &mat1[N*N*iter];
            float *B = &mat2[N*N*iter];
            float *C = &result[N*N*iter];

            // Zero C in DRAM
            for (int i = 0; i < N * N; i++) {
                C[i] = 0.0f;
            }
            bsg_fence(); // wait for zero-writes to reach DRAM

            // Tiled i-k-j loop
            for (int ii = 0; ii < N; ii += TILE_M) {
                int i_end = (ii + TILE_M < N) ? ii + TILE_M : N;

                for (int kk = 0; kk < N; kk += TILE_K) {
                    int k_end = (kk + TILE_K < N) ? kk + TILE_K : N;

                    for (int jj = 0; jj < N; jj += TILE_N) {
                        int j_end = (jj + TILE_N < N) ? jj + TILE_N : N;

                        // Micro-tiled rows (4 at a time)
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

            bsg_printf("[DBG] checkpoint 4: iter %d complete\n", iter);
        }

        bsg_printf("[DBG] checkpoint 5: all computation done\n");

        bsg_fence();

        bsg_printf("[DBG] checkpoint 6: fence done\n");
    }

    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 7: barrier done\n");

    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 8: returning\n");

    return 0;
}

