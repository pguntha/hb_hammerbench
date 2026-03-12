// Rev 2.6: Parallel SGEMM — interleaved fmaf() scheduling.
// Processes two k-values at once, alternating rows to break bypass stalls.

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X]={0};
int alert = 0;

#define DBL_K   8
#define DBL_N   16
#define MICRO_M 4
#define UNROLL_J 8

// Global DMEM buffers (not stack)
float a_buf[2][MICRO_M * DBL_K];
float b_buf[2][DBL_K * DBL_N];
float c_local[MICRO_M * DBL_N];

static inline void load_a_tile(float *A, float *a_dst,
                                int i_abs, int kk, int tile_k) {
    for (int mi = 0; mi < MICRO_M; mi++)
        for (int mk = 0; mk < tile_k; mk++)
            a_dst[mi * DBL_K + mk] = A[(N*(i_abs + mi)) + kk + mk];
}

static inline void load_b_tile(float *B, float *b_dst,
                                int kk, int tile_k,
                                int j_abs, int tile_n) {
    for (int mk = 0; mk < tile_k; mk++)
        for (int mj = 0; mj < tile_n; mj++)
            b_dst[mk * DBL_N + mj] = B[(N*(kk + mk)) + j_abs + mj];
}

// Interleaved FMA: k+=2, alternating rows 0&1 with 2&3 across even/odd k
static inline void compute_tile_interleaved(float *a_src, float *b_src,
                                             float *c_dst, int tile_k, int tile_n) {
    int k = 0;
    for (; k + 1 < tile_k; k += 2) {
        float a0_e = a_src[0*DBL_K + k];   float a0_o = a_src[0*DBL_K + k+1];
        float a1_e = a_src[1*DBL_K + k];   float a1_o = a_src[1*DBL_K + k+1];
        float a2_e = a_src[2*DBL_K + k];   float a2_o = a_src[2*DBL_K + k+1];
        float a3_e = a_src[3*DBL_K + k];   float a3_o = a_src[3*DBL_K + k+1];

        int j = 0;
        for (; j + UNROLL_J <= tile_n; j += UNROLL_J) {
            float be0=b_src[k*DBL_N+j+0],     be1=b_src[k*DBL_N+j+1];
            float be2=b_src[k*DBL_N+j+2],     be3=b_src[k*DBL_N+j+3];
            float be4=b_src[k*DBL_N+j+4],     be5=b_src[k*DBL_N+j+5];
            float be6=b_src[k*DBL_N+j+6],     be7=b_src[k*DBL_N+j+7];

            float bo0=b_src[(k+1)*DBL_N+j+0], bo1=b_src[(k+1)*DBL_N+j+1];
            float bo2=b_src[(k+1)*DBL_N+j+2], bo3=b_src[(k+1)*DBL_N+j+3];
            float bo4=b_src[(k+1)*DBL_N+j+4], bo5=b_src[(k+1)*DBL_N+j+5];
            float bo6=b_src[(k+1)*DBL_N+j+6], bo7=b_src[(k+1)*DBL_N+j+7];

            c_dst[0*DBL_N+j+0] = fmaf(a0_e, be0, c_dst[0*DBL_N+j+0]);
            c_dst[1*DBL_N+j+0] = fmaf(a1_e, be0, c_dst[1*DBL_N+j+0]);
            c_dst[0*DBL_N+j+0] = fmaf(a0_o, bo0, c_dst[0*DBL_N+j+0]);
            c_dst[1*DBL_N+j+0] = fmaf(a1_o, bo0, c_dst[1*DBL_N+j+0]);
            c_dst[2*DBL_N+j+0] = fmaf(a2_e, be0, c_dst[2*DBL_N+j+0]);
            c_dst[3*DBL_N+j+0] = fmaf(a3_e, be0, c_dst[3*DBL_N+j+0]);
            c_dst[2*DBL_N+j+0] = fmaf(a2_o, bo0, c_dst[2*DBL_N+j+0]);
            c_dst[3*DBL_N+j+0] = fmaf(a3_o, bo0, c_dst[3*DBL_N+j+0]);

            c_dst[0*DBL_N+j+1] = fmaf(a0_e, be1, c_dst[0*DBL_N+j+1]);
            c_dst[1*DBL_N+j+1] = fmaf(a1_e, be1, c_dst[1*DBL_N+j+1]);
            c_dst[0*DBL_N+j+1] = fmaf(a0_o, bo1, c_dst[0*DBL_N+j+1]);
            c_dst[1*DBL_N+j+1] = fmaf(a1_o, bo1, c_dst[1*DBL_N+j+1]);
            c_dst[2*DBL_N+j+1] = fmaf(a2_e, be1, c_dst[2*DBL_N+j+1]);
            c_dst[3*DBL_N+j+1] = fmaf(a3_e, be1, c_dst[3*DBL_N+j+1]);
            c_dst[2*DBL_N+j+1] = fmaf(a2_o, bo1, c_dst[2*DBL_N+j+1]);
            c_dst[3*DBL_N+j+1] = fmaf(a3_o, bo1, c_dst[3*DBL_N+j+1]);

            c_dst[0*DBL_N+j+2] = fmaf(a0_e, be2, c_dst[0*DBL_N+j+2]);
            c_dst[1*DBL_N+j+2] = fmaf(a1_e, be2, c_dst[1*DBL_N+j+2]);
            c_dst[0*DBL_N+j+2] = fmaf(a0_o, bo2, c_dst[0*DBL_N+j+2]);
            c_dst[1*DBL_N+j+2] = fmaf(a1_o, bo2, c_dst[1*DBL_N+j+2]);
            c_dst[2*DBL_N+j+2] = fmaf(a2_e, be2, c_dst[2*DBL_N+j+2]);
            c_dst[3*DBL_N+j+2] = fmaf(a3_e, be2, c_dst[3*DBL_N+j+2]);
            c_dst[2*DBL_N+j+2] = fmaf(a2_o, bo2, c_dst[2*DBL_N+j+2]);
            c_dst[3*DBL_N+j+2] = fmaf(a3_o, bo2, c_dst[3*DBL_N+j+2]);

            c_dst[0*DBL_N+j+3] = fmaf(a0_e, be3, c_dst[0*DBL_N+j+3]);
            c_dst[1*DBL_N+j+3] = fmaf(a1_e, be3, c_dst[1*DBL_N+j+3]);
            c_dst[0*DBL_N+j+3] = fmaf(a0_o, bo3, c_dst[0*DBL_N+j+3]);
            c_dst[1*DBL_N+j+3] = fmaf(a1_o, bo3, c_dst[1*DBL_N+j+3]);
            c_dst[2*DBL_N+j+3] = fmaf(a2_e, be3, c_dst[2*DBL_N+j+3]);
            c_dst[3*DBL_N+j+3] = fmaf(a3_e, be3, c_dst[3*DBL_N+j+3]);
            c_dst[2*DBL_N+j+3] = fmaf(a2_o, bo3, c_dst[2*DBL_N+j+3]);
            c_dst[3*DBL_N+j+3] = fmaf(a3_o, bo3, c_dst[3*DBL_N+j+3]);

            c_dst[0*DBL_N+j+4] = fmaf(a0_e, be4, c_dst[0*DBL_N+j+4]);
            c_dst[1*DBL_N+j+4] = fmaf(a1_e, be4, c_dst[1*DBL_N+j+4]);
            c_dst[0*DBL_N+j+4] = fmaf(a0_o, bo4, c_dst[0*DBL_N+j+4]);
            c_dst[1*DBL_N+j+4] = fmaf(a1_o, bo4, c_dst[1*DBL_N+j+4]);
            c_dst[2*DBL_N+j+4] = fmaf(a2_e, be4, c_dst[2*DBL_N+j+4]);
            c_dst[3*DBL_N+j+4] = fmaf(a3_e, be4, c_dst[3*DBL_N+j+4]);
            c_dst[2*DBL_N+j+4] = fmaf(a2_o, bo4, c_dst[2*DBL_N+j+4]);
            c_dst[3*DBL_N+j+4] = fmaf(a3_o, bo4, c_dst[3*DBL_N+j+4]);

            c_dst[0*DBL_N+j+5] = fmaf(a0_e, be5, c_dst[0*DBL_N+j+5]);
            c_dst[1*DBL_N+j+5] = fmaf(a1_e, be5, c_dst[1*DBL_N+j+5]);
            c_dst[0*DBL_N+j+5] = fmaf(a0_o, bo5, c_dst[0*DBL_N+j+5]);
            c_dst[1*DBL_N+j+5] = fmaf(a1_o, bo5, c_dst[1*DBL_N+j+5]);
            c_dst[2*DBL_N+j+5] = fmaf(a2_e, be5, c_dst[2*DBL_N+j+5]);
            c_dst[3*DBL_N+j+5] = fmaf(a3_e, be5, c_dst[3*DBL_N+j+5]);
            c_dst[2*DBL_N+j+5] = fmaf(a2_o, bo5, c_dst[2*DBL_N+j+5]);
            c_dst[3*DBL_N+j+5] = fmaf(a3_o, bo5, c_dst[3*DBL_N+j+5]);

            c_dst[0*DBL_N+j+6] = fmaf(a0_e, be6, c_dst[0*DBL_N+j+6]);
            c_dst[1*DBL_N+j+6] = fmaf(a1_e, be6, c_dst[1*DBL_N+j+6]);
            c_dst[0*DBL_N+j+6] = fmaf(a0_o, bo6, c_dst[0*DBL_N+j+6]);
            c_dst[1*DBL_N+j+6] = fmaf(a1_o, bo6, c_dst[1*DBL_N+j+6]);
            c_dst[2*DBL_N+j+6] = fmaf(a2_e, be6, c_dst[2*DBL_N+j+6]);
            c_dst[3*DBL_N+j+6] = fmaf(a3_e, be6, c_dst[3*DBL_N+j+6]);
            c_dst[2*DBL_N+j+6] = fmaf(a2_o, bo6, c_dst[2*DBL_N+j+6]);
            c_dst[3*DBL_N+j+6] = fmaf(a3_o, bo6, c_dst[3*DBL_N+j+6]);

            c_dst[0*DBL_N+j+7] = fmaf(a0_e, be7, c_dst[0*DBL_N+j+7]);
            c_dst[1*DBL_N+j+7] = fmaf(a1_e, be7, c_dst[1*DBL_N+j+7]);
            c_dst[0*DBL_N+j+7] = fmaf(a0_o, bo7, c_dst[0*DBL_N+j+7]);
            c_dst[1*DBL_N+j+7] = fmaf(a1_o, bo7, c_dst[1*DBL_N+j+7]);
            c_dst[2*DBL_N+j+7] = fmaf(a2_e, be7, c_dst[2*DBL_N+j+7]);
            c_dst[3*DBL_N+j+7] = fmaf(a3_e, be7, c_dst[3*DBL_N+j+7]);
            c_dst[2*DBL_N+j+7] = fmaf(a2_o, bo7, c_dst[2*DBL_N+j+7]);
            c_dst[3*DBL_N+j+7] = fmaf(a3_o, bo7, c_dst[3*DBL_N+j+7]);
        }
        for (; j < tile_n; j++) {
            float bve = b_src[k*DBL_N+j], bvo = b_src[(k+1)*DBL_N+j];
            c_dst[0*DBL_N+j] = fmaf(a0_o, bvo, fmaf(a0_e, bve, c_dst[0*DBL_N+j]));
            c_dst[1*DBL_N+j] = fmaf(a1_o, bvo, fmaf(a1_e, bve, c_dst[1*DBL_N+j]));
            c_dst[2*DBL_N+j] = fmaf(a2_o, bvo, fmaf(a2_e, bve, c_dst[2*DBL_N+j]));
            c_dst[3*DBL_N+j] = fmaf(a3_o, bvo, fmaf(a3_e, bve, c_dst[3*DBL_N+j]));
        }
    }
    for (; k < tile_k; k++) {
        float a0 = a_src[0*DBL_K+k], a1 = a_src[1*DBL_K+k];
        float a2 = a_src[2*DBL_K+k], a3 = a_src[3*DBL_K+k];
        for (int j = 0; j < tile_n; j++) {
            float bv = b_src[k*DBL_N+j];
            c_dst[0*DBL_N+j] = fmaf(a0, bv, c_dst[0*DBL_N+j]);
            c_dst[1*DBL_N+j] = fmaf(a1, bv, c_dst[1*DBL_N+j]);
            c_dst[2*DBL_N+j] = fmaf(a2, bv, c_dst[2*DBL_N+j]);
            c_dst[3*DBL_N+j] = fmaf(a3, bv, c_dst[3*DBL_N+j]);
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
    int num_k_tiles = (N + DBL_K - 1) / DBL_K;
    int k_tile_offset = __bsg_id % num_k_tiles;

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
                    int ke = (kk+DBL_K<N)?kk+DBL_K:N;
                    for (int ri = 0; ri < i_cnt; ri++)
                        for (int k = kk; k < ke; k++) {
                            float av = A[(N*(i_abs+ri))+k];
                            for (int j = col_start; j < col_end; j++)
                                C[(N*(i_abs+ri))+j] = fmaf(av, B[(N*k)+j], C[(N*(i_abs+ri))+j]);
                        }
                }
                continue;
            }

            for (int jj = 0; jj < my_N; jj += DBL_N) {
                int j_abs = col_start + jj;
                int j_end_abs = (j_abs+DBL_N < col_end) ? j_abs+DBL_N : col_end;
                int tile_n = j_end_abs - j_abs;

                for (int mi = 0; mi < MICRO_M; mi++)
                    for (int mj = 0; mj < tile_n; mj++)
                        c_local[mi*DBL_N+mj] = 0.0f;

                int cur = 0, nxt = 1;
                int kt_first = k_tile_offset;
                int kk_first = kt_first * DBL_K;
                int tk_first = (kk_first + DBL_K < N) ? DBL_K : N - kk_first;
                load_a_tile(A, a_buf[cur], i_abs, kk_first, tk_first);
                load_b_tile(B, b_buf[cur], kk_first, tk_first, j_abs, tile_n);

                for (int s = 0; s < num_k_tiles; s++) {
                    int kt_cur = (k_tile_offset + s) % num_k_tiles;
                    int kk_cur = kt_cur * DBL_K;
                    int tk_cur = (kk_cur + DBL_K < N) ? DBL_K : N - kk_cur;

                    if (s + 1 < num_k_tiles) {
                        int kt_nxt = (k_tile_offset + s + 1) % num_k_tiles;
                        int kk_nxt = kt_nxt * DBL_K;
                        int tk_nxt = (kk_nxt + DBL_K < N) ? DBL_K : N - kk_nxt;
                        load_a_tile(A, a_buf[nxt], i_abs, kk_nxt, tk_nxt);
                        load_b_tile(B, b_buf[nxt], kk_nxt, tk_nxt, j_abs, tile_n);
                    }

                    compute_tile_interleaved(a_buf[cur], b_buf[cur],
                                              c_local, tk_cur, tile_n);

                    int tmp = cur; cur = nxt; nxt = tmp;
                }

                // Write C sub-tile DMEM → DRAM
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

