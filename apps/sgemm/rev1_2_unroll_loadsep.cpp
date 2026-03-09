// Rev 1.2: Serial SGEMM — j-loop unrolled by 8 with load/compute/store separation.
// Groups non-blocking DRAM loads before FMAs to create load-use distance.
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
#define UNROLL_J 8

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
        bsg_printf("[DBG] checkpoint 3: N=%d NITER=%d TILE_M=%d TILE_K=%d TILE_N=%d MICRO_M=%d UNROLL_J=%d\n",
                   N, NITER, TILE_M, TILE_K, TILE_N, MICRO_M, UNROLL_J);

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

                                // Unrolled j-loop with load/compute/store separation
                                int j = jj;
                                for (; j + UNROLL_J <= j_end; j += UNROLL_J) {

                                    // Load B and C (non-blocking DRAM reads)
                                    float b0 = B[(N*k) + j + 0];
                                    float b1 = B[(N*k) + j + 1];
                                    float b2 = B[(N*k) + j + 2];
                                    float b3 = B[(N*k) + j + 3];
                                    float b4 = B[(N*k) + j + 4];
                                    float b5 = B[(N*k) + j + 5];
                                    float b6 = B[(N*k) + j + 6];
                                    float b7 = B[(N*k) + j + 7];

                                    float c00 = C[(N*(i+0)) + j+0]; float c01 = C[(N*(i+0)) + j+1];
                                    float c02 = C[(N*(i+0)) + j+2]; float c03 = C[(N*(i+0)) + j+3];
                                    float c04 = C[(N*(i+0)) + j+4]; float c05 = C[(N*(i+0)) + j+5];
                                    float c06 = C[(N*(i+0)) + j+6]; float c07 = C[(N*(i+0)) + j+7];

                                    float c10 = C[(N*(i+1)) + j+0]; float c11 = C[(N*(i+1)) + j+1];
                                    float c12 = C[(N*(i+1)) + j+2]; float c13 = C[(N*(i+1)) + j+3];
                                    float c14 = C[(N*(i+1)) + j+4]; float c15 = C[(N*(i+1)) + j+5];
                                    float c16 = C[(N*(i+1)) + j+6]; float c17 = C[(N*(i+1)) + j+7];

                                    float c20 = C[(N*(i+2)) + j+0]; float c21 = C[(N*(i+2)) + j+1];
                                    float c22 = C[(N*(i+2)) + j+2]; float c23 = C[(N*(i+2)) + j+3];
                                    float c24 = C[(N*(i+2)) + j+4]; float c25 = C[(N*(i+2)) + j+5];
                                    float c26 = C[(N*(i+2)) + j+6]; float c27 = C[(N*(i+2)) + j+7];

                                    float c30 = C[(N*(i+3)) + j+0]; float c31 = C[(N*(i+3)) + j+1];
                                    float c32 = C[(N*(i+3)) + j+2]; float c33 = C[(N*(i+3)) + j+3];
                                    float c34 = C[(N*(i+3)) + j+4]; float c35 = C[(N*(i+3)) + j+5];
                                    float c36 = C[(N*(i+3)) + j+6]; float c37 = C[(N*(i+3)) + j+7];

                                    asm volatile ("" ::: "memory");

                                    // FMAs
                                    c00 += a0 * b0; c01 += a0 * b1;
                                    c02 += a0 * b2; c03 += a0 * b3;
                                    c04 += a0 * b4; c05 += a0 * b5;
                                    c06 += a0 * b6; c07 += a0 * b7;

                                    c10 += a1 * b0; c11 += a1 * b1;
                                    c12 += a1 * b2; c13 += a1 * b3;
                                    c14 += a1 * b4; c15 += a1 * b5;
                                    c16 += a1 * b6; c17 += a1 * b7;

                                    c20 += a2 * b0; c21 += a2 * b1;
                                    c22 += a2 * b2; c23 += a2 * b3;
                                    c24 += a2 * b4; c25 += a2 * b5;
                                    c26 += a2 * b6; c27 += a2 * b7;

                                    c30 += a3 * b0; c31 += a3 * b1;
                                    c32 += a3 * b2; c33 += a3 * b3;
                                    c34 += a3 * b4; c35 += a3 * b5;
                                    c36 += a3 * b6; c37 += a3 * b7;

                                    asm volatile ("" ::: "memory");

                                    // Store C back
                                    C[(N*(i+0)) + j+0] = c00; C[(N*(i+0)) + j+1] = c01;
                                    C[(N*(i+0)) + j+2] = c02; C[(N*(i+0)) + j+3] = c03;
                                    C[(N*(i+0)) + j+4] = c04; C[(N*(i+0)) + j+5] = c05;
                                    C[(N*(i+0)) + j+6] = c06; C[(N*(i+0)) + j+7] = c07;

                                    C[(N*(i+1)) + j+0] = c10; C[(N*(i+1)) + j+1] = c11;
                                    C[(N*(i+1)) + j+2] = c12; C[(N*(i+1)) + j+3] = c13;
                                    C[(N*(i+1)) + j+4] = c14; C[(N*(i+1)) + j+5] = c15;
                                    C[(N*(i+1)) + j+6] = c16; C[(N*(i+1)) + j+7] = c17;

                                    C[(N*(i+2)) + j+0] = c20; C[(N*(i+2)) + j+1] = c21;
                                    C[(N*(i+2)) + j+2] = c22; C[(N*(i+2)) + j+3] = c23;
                                    C[(N*(i+2)) + j+4] = c24; C[(N*(i+2)) + j+5] = c25;
                                    C[(N*(i+2)) + j+6] = c26; C[(N*(i+2)) + j+7] = c27;

                                    C[(N*(i+3)) + j+0] = c30; C[(N*(i+3)) + j+1] = c31;
                                    C[(N*(i+3)) + j+2] = c32; C[(N*(i+3)) + j+3] = c33;
                                    C[(N*(i+3)) + j+4] = c34; C[(N*(i+3)) + j+5] = c35;
                                    C[(N*(i+3)) + j+6] = c36; C[(N*(i+3)) + j+7] = c37;
                                }

                                for (; j < j_end; j++) {
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

