// Rev 1.3: Serial SGEMM — A and C tiles buffered in DMEM scratchpad.
// A reads are local (single-cycle); B still from DRAM. Final serial baseline.
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

        float a_local[MICRO_M * TILE_K];
        float c_local[MICRO_M * TILE_N];

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

            // Tiled loop over rows and columns
            for (int ii = 0; ii < N; ii += TILE_M) {
                int i_end = (ii + TILE_M < N) ? ii + TILE_M : N;

                for (int jj = 0; jj < N; jj += TILE_N) {
                    int j_end = (jj + TILE_N < N) ? jj + TILE_N : N;
                    int tile_n = j_end - jj;

                    // Micro-tiled rows (4 at a time)
                    int i = ii;
                    for (; i + MICRO_M <= i_end; i += MICRO_M) {

                        // Load C sub-tile from DRAM into DMEM accumulator
                        for (int mi = 0; mi < MICRO_M; mi++) {
                            for (int mj = 0; mj < tile_n; mj++) {
                                c_local[mi * TILE_N + mj] = C[(N*(i + mi)) + jj + mj];
                            }
                        }

                        for (int kk = 0; kk < N; kk += TILE_K) {
                            int k_end = (kk + TILE_K < N) ? kk + TILE_K : N;
                            int tile_k = k_end - kk;

                            // Copy A micro-tile DRAM → DMEM
                            for (int mi = 0; mi < MICRO_M; mi++) {
                                for (int mk = 0; mk < tile_k; mk++) {
                                    a_local[mi * TILE_K + mk] = A[(N*(i + mi)) + kk + mk];
                                }
                            }

                            // A from DMEM, B from DRAM, accumulate into DMEM c_local
                            for (int k = 0; k < tile_k; k++) {
                                float a0 = a_local[0 * TILE_K + k];
                                float a1 = a_local[1 * TILE_K + k];
                                float a2 = a_local[2 * TILE_K + k];
                                float a3 = a_local[3 * TILE_K + k];

                                int j = 0;
                                for (; j + UNROLL_J <= tile_n; j += UNROLL_J) {
                                    float b0 = B[(N*(kk + k)) + jj + j + 0];
                                    float b1 = B[(N*(kk + k)) + jj + j + 1];
                                    float b2 = B[(N*(kk + k)) + jj + j + 2];
                                    float b3 = B[(N*(kk + k)) + jj + j + 3];
                                    float b4 = B[(N*(kk + k)) + jj + j + 4];
                                    float b5 = B[(N*(kk + k)) + jj + j + 5];
                                    float b6 = B[(N*(kk + k)) + jj + j + 6];
                                    float b7 = B[(N*(kk + k)) + jj + j + 7];

                                    asm volatile ("" ::: "memory");

                                    c_local[0*TILE_N+j+0] += a0*b0; c_local[0*TILE_N+j+1] += a0*b1;
                                    c_local[0*TILE_N+j+2] += a0*b2; c_local[0*TILE_N+j+3] += a0*b3;
                                    c_local[0*TILE_N+j+4] += a0*b4; c_local[0*TILE_N+j+5] += a0*b5;
                                    c_local[0*TILE_N+j+6] += a0*b6; c_local[0*TILE_N+j+7] += a0*b7;

                                    c_local[1*TILE_N+j+0] += a1*b0; c_local[1*TILE_N+j+1] += a1*b1;
                                    c_local[1*TILE_N+j+2] += a1*b2; c_local[1*TILE_N+j+3] += a1*b3;
                                    c_local[1*TILE_N+j+4] += a1*b4; c_local[1*TILE_N+j+5] += a1*b5;
                                    c_local[1*TILE_N+j+6] += a1*b6; c_local[1*TILE_N+j+7] += a1*b7;

                                    c_local[2*TILE_N+j+0] += a2*b0; c_local[2*TILE_N+j+1] += a2*b1;
                                    c_local[2*TILE_N+j+2] += a2*b2; c_local[2*TILE_N+j+3] += a2*b3;
                                    c_local[2*TILE_N+j+4] += a2*b4; c_local[2*TILE_N+j+5] += a2*b5;
                                    c_local[2*TILE_N+j+6] += a2*b6; c_local[2*TILE_N+j+7] += a2*b7;

                                    c_local[3*TILE_N+j+0] += a3*b0; c_local[3*TILE_N+j+1] += a3*b1;
                                    c_local[3*TILE_N+j+2] += a3*b2; c_local[3*TILE_N+j+3] += a3*b3;
                                    c_local[3*TILE_N+j+4] += a3*b4; c_local[3*TILE_N+j+5] += a3*b5;
                                    c_local[3*TILE_N+j+6] += a3*b6; c_local[3*TILE_N+j+7] += a3*b7;
                                }

                                for (; j < tile_n; j++) {
                                    float b_val = B[(N*(kk + k)) + jj + j];
                                    c_local[0*TILE_N+j] += a0 * b_val;
                                    c_local[1*TILE_N+j] += a1 * b_val;
                                    c_local[2*TILE_N+j] += a2 * b_val;
                                    c_local[3*TILE_N+j] += a3 * b_val;
                                }
                            }
                        }

                        // Write C tile DMEM → DRAM
                        for (int mi = 0; mi < MICRO_M; mi++) {
                            for (int mj = 0; mj < tile_n; mj++) {
                                C[(N*(i + mi)) + jj + mj] = c_local[mi * TILE_N + mj];
                            }
                        }
                    }

                    // Remainder rows
                    for (; i < i_end; i++) {
                        for (int kk = 0; kk < N; kk += TILE_K) {
                            int k_end = (kk + TILE_K < N) ? kk + TILE_K : N;
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

