// Rev 2.4: Parallel SGEMM — double-buffered A/B to overlap loads with compute.
// While computing on buf[cur], next k-tile is loaded into buf[nxt].

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X] = {0};
int alert = 0;

#define DBL_K    8
#define DBL_N    16
#define MICRO_M  4
#define UNROLL_J 8

// Global DMEM buffers (not stack)
float a_buf[2][MICRO_M * DBL_K];
float b_buf[2][DBL_K  * DBL_N];
float c_local[MICRO_M * DBL_N];

static inline void load_a_tile(float *A, float *dst,
                                int i_abs, int kk, int tile_k) {
    for (int mi = 0; mi < MICRO_M; mi++)
        for (int mk = 0; mk < tile_k; mk++)
            dst[mi*DBL_K+mk] = A[(N*(i_abs+mi)) + kk+mk];
}

static inline void load_b_tile(float *B, float *dst,
                                int kk, int tile_k, int j_abs, int tile_n) {
    for (int mk = 0; mk < tile_k; mk++)
        for (int mj = 0; mj < tile_n; mj++)
            dst[mk*DBL_N+mj] = B[(N*(kk+mk)) + j_abs+mj];
}

static inline void compute_tile(float *as, float *bs,
                                 float *cs, int tile_k, int tile_n) {
    for (int k = 0; k < tile_k; k++) {
        float a0=as[0*DBL_K+k], a1=as[1*DBL_K+k];
        float a2=as[2*DBL_K+k], a3=as[3*DBL_K+k];

        int j = 0;
        for (; j + UNROLL_J <= tile_n; j += UNROLL_J) {
            float b0=bs[k*DBL_N+j+0], b1=bs[k*DBL_N+j+1];
            float b2=bs[k*DBL_N+j+2], b3=bs[k*DBL_N+j+3];
            float b4=bs[k*DBL_N+j+4], b5=bs[k*DBL_N+j+5];
            float b6=bs[k*DBL_N+j+6], b7=bs[k*DBL_N+j+7];

            cs[0*DBL_N+j+0]+=a0*b0; cs[0*DBL_N+j+1]+=a0*b1;
            cs[0*DBL_N+j+2]+=a0*b2; cs[0*DBL_N+j+3]+=a0*b3;
            cs[0*DBL_N+j+4]+=a0*b4; cs[0*DBL_N+j+5]+=a0*b5;
            cs[0*DBL_N+j+6]+=a0*b6; cs[0*DBL_N+j+7]+=a0*b7;

            cs[1*DBL_N+j+0]+=a1*b0; cs[1*DBL_N+j+1]+=a1*b1;
            cs[1*DBL_N+j+2]+=a1*b2; cs[1*DBL_N+j+3]+=a1*b3;
            cs[1*DBL_N+j+4]+=a1*b4; cs[1*DBL_N+j+5]+=a1*b5;
            cs[1*DBL_N+j+6]+=a1*b6; cs[1*DBL_N+j+7]+=a1*b7;

            cs[2*DBL_N+j+0]+=a2*b0; cs[2*DBL_N+j+1]+=a2*b1;
            cs[2*DBL_N+j+2]+=a2*b2; cs[2*DBL_N+j+3]+=a2*b3;
            cs[2*DBL_N+j+4]+=a2*b4; cs[2*DBL_N+j+5]+=a2*b5;
            cs[2*DBL_N+j+6]+=a2*b6; cs[2*DBL_N+j+7]+=a2*b7;

            cs[3*DBL_N+j+0]+=a3*b0; cs[3*DBL_N+j+1]+=a3*b1;
            cs[3*DBL_N+j+2]+=a3*b2; cs[3*DBL_N+j+3]+=a3*b3;
            cs[3*DBL_N+j+4]+=a3*b4; cs[3*DBL_N+j+5]+=a3*b5;
            cs[3*DBL_N+j+6]+=a3*b6; cs[3*DBL_N+j+7]+=a3*b7;
        }
        for (; j < tile_n; j++) {
            float bv = bs[k*DBL_N+j];
            cs[0*DBL_N+j]+=a0*bv; cs[1*DBL_N+j]+=a1*bv;
            cs[2*DBL_N+j]+=a2*bv; cs[3*DBL_N+j]+=a3*bv;
        }
    }
}

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
    int num_k_tiles   = (N + DBL_K - 1) / DBL_K;

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 3: P_r=%d P_c=%d rpc=%d cpc=%d nkt=%d\n",
                                   P_r, P_c, rows_per_core, cols_per_core, num_k_tiles);

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
                for (int kk = 0; kk < N; kk += DBL_K) {
                    int ke = (kk+DBL_K<N) ? kk+DBL_K : N;
                    for (int ri = 0; ri < i_cnt; ri++)
                        for (int k = kk; k < ke; k++) {
                            float av = A[(N*(i_abs+ri))+k];
                            for (int j = col_start; j < col_end; j++)
                                C[(N*(i_abs+ri))+j] += av*B[(N*k)+j];
                        }
                }
                continue;
            }

            for (int jj = 0; jj < my_N; jj += DBL_N) {
                int j_abs     = col_start + jj;
                int j_end_abs = (j_abs+DBL_N < col_end) ? j_abs+DBL_N : col_end;
                int tile_n    = j_end_abs - j_abs;

                // Load C sub-tile DRAM → DMEM
                for (int mi = 0; mi < MICRO_M; mi++)
                    for (int mj = 0; mj < tile_n; mj++)
                        c_local[mi*DBL_N+mj] = C[(N*(i_abs+mi)) + j_abs+mj];

                // Double-buffered k-tile loop
                int cur = 0, nxt = 1;

                // Preload first k-tile
                int kk0 = 0;
                int tk0 = (DBL_K < N) ? DBL_K : N;
                load_a_tile(A, a_buf[cur], i_abs, kk0, tk0);
                load_b_tile(B, b_buf[cur], kk0, tk0, j_abs, tile_n);

                for (int kt = 0; kt < num_k_tiles; kt++) {
                    int kk_cur = kt * DBL_K;
                    int tk_cur = (kk_cur+DBL_K < N) ? DBL_K : N-kk_cur;

                    // Prefetch next k-tile into buf[nxt]
                    if (kt + 1 < num_k_tiles) {
                        int kk_nxt = (kt+1) * DBL_K;
                        int tk_nxt = (kk_nxt+DBL_K < N) ? DBL_K : N-kk_nxt;
                        load_a_tile(A, a_buf[nxt], i_abs, kk_nxt, tk_nxt);
                        load_b_tile(B, b_buf[nxt], kk_nxt, tk_nxt, j_abs, tile_n);
                    }

                    compute_tile(a_buf[cur], b_buf[cur], c_local, tk_cur, tile_n);

                    int tmp = cur; cur = nxt; nxt = tmp;
                }

                // Write C tile DMEM → DRAM
                for (int mi = 0; mi < MICRO_M; mi++)
                    for (int mj = 0; mj < tile_n; mj++)
                        C[(N*(i_abs+mi)) + j_abs+mj] = c_local[mi*DBL_N+mj];
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

