// Rev 2.2: Parallel SGEMM — A, B, and C all buffered in DMEM scratchpad.
// All inner-loop reads are local; zero network traffic during compute.

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X] = {0};
int alert = 0;

#define BLK_K    16
#define BLK_N    16
#define MICRO_M  4
#define UNROLL_J 8

// Global DMEM buffers (not stack)
float a_local[MICRO_M * BLK_K];
float b_local[BLK_K  * BLK_N];
float c_local[MICRO_M * BLK_N];

extern "C"
int kernel(float *mat1, float *mat2, float *result, int pod_id)
{
    bsg_barrier_tile_group_init();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 1: barrier_init done\n");

    bsg_barrier_multipod(pod_id, NUM_POD_X, done, &alert);

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 2: multipod barrier done\n");

    bsg_cuda_print_stat_kernel_start();

    int P_r = bsg_tiles_Y;
    int P_c = bsg_tiles_X;
    int rows_per_core = (N + P_r - 1) / P_r;
    int cols_per_core = (N + P_c - 1) / P_c;

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 3: P_r=%d P_c=%d rpc=%d cpc=%d\n",
                                   P_r, P_c, rows_per_core, cols_per_core);

    bsg_unroll(1)
    for (int iter = 0; iter < NITER; iter++) {
        float *A = &mat1[N*N*iter];
        float *B = &mat2[N*N*iter];
        float *C = &result[N*N*iter];

        int row_start = __bsg_y * rows_per_core;
        int row_end   = (row_start + rows_per_core < N) ? row_start + rows_per_core : N;
        int col_start = __bsg_x * cols_per_core;
        int col_end   = (col_start + cols_per_core < N) ? col_start + cols_per_core : N;
        int my_M = row_end - row_start;
        int my_N = col_end - col_start;

        // Zero this core's C region in DRAM
        for (int i = row_start; i < row_end; i++)
            for (int j = col_start; j < col_end; j++)
                C[(N*i) + j] = 0.0f;

        bsg_fence(); // wait for zero-writes to reach DRAM

        // Micro-tiled row loop
        for (int ii = 0; ii < my_M; ii += MICRO_M) {
            int i_cnt = ((ii + MICRO_M) <= my_M) ? MICRO_M : (my_M - ii);
            int i_abs = row_start + ii;

            // Scalar fallback for remainder rows
            if (i_cnt < MICRO_M) {
                for (int kk = 0; kk < N; kk += BLK_K) {
                    int ke = (kk + BLK_K < N) ? kk + BLK_K : N;
                    for (int ri = 0; ri < i_cnt; ri++)
                        for (int k = kk; k < ke; k++) {
                            float av = A[(N*(i_abs+ri))+k];
                            for (int j = col_start; j < col_end; j++)
                                C[(N*(i_abs+ri))+j] += av * B[(N*k)+j];
                        }
                }
                continue;
            }

            for (int jj = 0; jj < my_N; jj += BLK_N) {
                int j_abs     = col_start + jj;
                int j_end_abs = (j_abs + BLK_N < col_end) ? j_abs + BLK_N : col_end;
                int tile_n    = j_end_abs - j_abs;

                // Load C sub-tile DRAM → DMEM
                for (int mi = 0; mi < MICRO_M; mi++)
                    for (int mj = 0; mj < tile_n; mj++)
                        c_local[mi*BLK_N+mj] = C[(N*(i_abs+mi)) + j_abs+mj];

                for (int kk = 0; kk < N; kk += BLK_K) {
                    int k_end  = (kk + BLK_K < N) ? kk + BLK_K : N;
                    int tile_k = k_end - kk;

                    // Copy A DRAM → DMEM
                    for (int mi = 0; mi < MICRO_M; mi++)
                        for (int mk = 0; mk < tile_k; mk++)
                            a_local[mi*BLK_K+mk] = A[(N*(i_abs+mi)) + kk+mk];

                    // Copy B DRAM → DMEM
                    for (int mk = 0; mk < tile_k; mk++)
                        for (int mj = 0; mj < tile_n; mj++)
                            b_local[mk*BLK_N+mj] = B[(N*(kk+mk)) + j_abs+mj];

                    // All reads from DMEM
                    for (int k = 0; k < tile_k; k++) {
                        float a0 = a_local[0*BLK_K+k];
                        float a1 = a_local[1*BLK_K+k];
                        float a2 = a_local[2*BLK_K+k];
                        float a3 = a_local[3*BLK_K+k];

                        int j = 0;
                        for (; j + UNROLL_J <= tile_n; j += UNROLL_J) {
                            float b0=b_local[k*BLK_N+j+0], b1=b_local[k*BLK_N+j+1];
                            float b2=b_local[k*BLK_N+j+2], b3=b_local[k*BLK_N+j+3];
                            float b4=b_local[k*BLK_N+j+4], b5=b_local[k*BLK_N+j+5];
                            float b6=b_local[k*BLK_N+j+6], b7=b_local[k*BLK_N+j+7];

                            c_local[0*BLK_N+j+0]+=a0*b0; c_local[0*BLK_N+j+1]+=a0*b1;
                            c_local[0*BLK_N+j+2]+=a0*b2; c_local[0*BLK_N+j+3]+=a0*b3;
                            c_local[0*BLK_N+j+4]+=a0*b4; c_local[0*BLK_N+j+5]+=a0*b5;
                            c_local[0*BLK_N+j+6]+=a0*b6; c_local[0*BLK_N+j+7]+=a0*b7;

                            c_local[1*BLK_N+j+0]+=a1*b0; c_local[1*BLK_N+j+1]+=a1*b1;
                            c_local[1*BLK_N+j+2]+=a1*b2; c_local[1*BLK_N+j+3]+=a1*b3;
                            c_local[1*BLK_N+j+4]+=a1*b4; c_local[1*BLK_N+j+5]+=a1*b5;
                            c_local[1*BLK_N+j+6]+=a1*b6; c_local[1*BLK_N+j+7]+=a1*b7;

                            c_local[2*BLK_N+j+0]+=a2*b0; c_local[2*BLK_N+j+1]+=a2*b1;
                            c_local[2*BLK_N+j+2]+=a2*b2; c_local[2*BLK_N+j+3]+=a2*b3;
                            c_local[2*BLK_N+j+4]+=a2*b4; c_local[2*BLK_N+j+5]+=a2*b5;
                            c_local[2*BLK_N+j+6]+=a2*b6; c_local[2*BLK_N+j+7]+=a2*b7;

                            c_local[3*BLK_N+j+0]+=a3*b0; c_local[3*BLK_N+j+1]+=a3*b1;
                            c_local[3*BLK_N+j+2]+=a3*b2; c_local[3*BLK_N+j+3]+=a3*b3;
                            c_local[3*BLK_N+j+4]+=a3*b4; c_local[3*BLK_N+j+5]+=a3*b5;
                            c_local[3*BLK_N+j+6]+=a3*b6; c_local[3*BLK_N+j+7]+=a3*b7;
                        }
                        for (; j < tile_n; j++) {
                            float bv = b_local[k*BLK_N+j];
                            c_local[0*BLK_N+j]+=a0*bv; c_local[1*BLK_N+j]+=a1*bv;
                            c_local[2*BLK_N+j]+=a2*bv; c_local[3*BLK_N+j]+=a3*bv;
                        }
                    }
                }

                // Write C tile DMEM → DRAM
                for (int mi = 0; mi < MICRO_M; mi++)
                    for (int mj = 0; mj < tile_n; mj++)
                        C[(N*(i_abs+mi)) + j_abs+mj] = c_local[mi*BLK_N+mj];
            }
        }

        if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 4: iter %d complete\n", iter);
    }

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 5: all computation done\n");

    bsg_fence();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 6: fence done\n");

    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 7: barrier done\n");

    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 8: returning\n");

    return 0;
}

