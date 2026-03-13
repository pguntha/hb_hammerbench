// Rev 2.8: Parallel SGEMM — simplified register-accumulation.
// Combines rev2_1's simple sequential tiling (no stagger, no double-buffer)
// with rev2_7's register accumulators and interleaved fmaf().

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X] = {0};
int alert = 0;

#define BLK_K    16
#define MICRO_M  4
#define SUB_N    4

// DMEM: only A and B staging; no c_local needed
float a_buf[MICRO_M * BLK_K];
float b_buf[BLK_K   * SUB_N];

// Register-staged A load: DRAM → register temps → DMEM
static inline void load_a_tile(float *A, int i_abs, int kk, int tile_k) {
    for (int mi = 0; mi < MICRO_M; mi++) {
        float *src = &A[N * (i_abs + mi) + kk];
        float *dst = &a_buf[mi * BLK_K];
        for (int mk = 0; mk < tile_k; mk++) {
            register float tmp = src[mk];
            asm volatile ("" ::: "memory");
            dst[mk] = tmp;
        }
    }
}

// Register-staged B load: only SUB_N=4 columns per strip
static inline void load_b_strip(float *B, int kk, int tile_k, int j_abs) {
    for (int mk = 0; mk < tile_k; mk++) {
        float *src = &B[N * (kk + mk) + j_abs];
        float *dst = &b_buf[mk * SUB_N];
        register float t0 = src[0];
        register float t1 = src[1];
        register float t2 = src[2];
        register float t3 = src[3];
        asm volatile ("" ::: "memory");
        dst[0] = t0; dst[1] = t1; dst[2] = t2; dst[3] = t3;
    }
}

// Computes MICRO_M x my_N block of C with register accumulators.
// js-strip (outer) x k-tile (inner): c00..c33 stay in registers throughout.
static inline void compute_c_tile(
    float *A, float *B, float *C,
    int i_abs, int col_start, int my_N)
{
    bsg_unroll(1)
    for (int js = 0; js + SUB_N <= my_N; js += SUB_N) {
        int j_abs = col_start + js;

        register float c00=0,c01=0,c02=0,c03=0;
        register float c10=0,c11=0,c12=0,c13=0;
        register float c20=0,c21=0,c22=0,c23=0;
        register float c30=0,c31=0,c32=0,c33=0;

        bsg_unroll(1)
        for (int kk = 0; kk < N; kk += BLK_K) {
            int tile_k = (kk + BLK_K < N) ? BLK_K : N - kk;

            load_a_tile(A, i_abs, kk, tile_k);
            load_b_strip(B, kk, tile_k, j_abs);

            // Interleaved k+=2 FMA
            int k = 0;
            for (; k + 1 < tile_k; k += 2) {
                register float a0e = a_buf[0*BLK_K+k],   a0o = a_buf[0*BLK_K+k+1];
                register float a1e = a_buf[1*BLK_K+k],   a1o = a_buf[1*BLK_K+k+1];
                register float a2e = a_buf[2*BLK_K+k],   a2o = a_buf[2*BLK_K+k+1];
                register float a3e = a_buf[3*BLK_K+k],   a3o = a_buf[3*BLK_K+k+1];

                register float be0 = b_buf[k    *SUB_N+0], be1 = b_buf[k    *SUB_N+1];
                register float be2 = b_buf[k    *SUB_N+2], be3 = b_buf[k    *SUB_N+3];
                register float bo0 = b_buf[(k+1)*SUB_N+0], bo1 = b_buf[(k+1)*SUB_N+1];
                register float bo2 = b_buf[(k+1)*SUB_N+2], bo3 = b_buf[(k+1)*SUB_N+3];

                asm volatile ("" ::: "memory");

                c00 = fmaf(a0e,be0,c00); c01 = fmaf(a0e,be1,c01);
                c10 = fmaf(a1e,be0,c10); c11 = fmaf(a1e,be1,c11);
                c00 = fmaf(a0o,bo0,c00); c01 = fmaf(a0o,bo1,c01);
                c10 = fmaf(a1o,bo0,c10); c11 = fmaf(a1o,bo1,c11);

                c20 = fmaf(a2e,be0,c20); c21 = fmaf(a2e,be1,c21);
                c30 = fmaf(a3e,be0,c30); c31 = fmaf(a3e,be1,c31);
                c20 = fmaf(a2o,bo0,c20); c21 = fmaf(a2o,bo1,c21);
                c30 = fmaf(a3o,bo0,c30); c31 = fmaf(a3o,bo1,c31);

                c02 = fmaf(a0e,be2,c02); c03 = fmaf(a0e,be3,c03);
                c12 = fmaf(a1e,be2,c12); c13 = fmaf(a1e,be3,c13);
                c02 = fmaf(a0o,bo2,c02); c03 = fmaf(a0o,bo3,c03);
                c12 = fmaf(a1o,bo2,c12); c13 = fmaf(a1o,bo3,c13);

                c22 = fmaf(a2e,be2,c22); c23 = fmaf(a2e,be3,c23);
                c32 = fmaf(a3e,be2,c32); c33 = fmaf(a3e,be3,c33);
                c22 = fmaf(a2o,bo2,c22); c23 = fmaf(a2o,bo3,c23);
                c32 = fmaf(a3o,bo2,c32); c33 = fmaf(a3o,bo3,c33);
            }

            for (; k < tile_k; k++) {
                register float a0 = a_buf[0*BLK_K+k], a1 = a_buf[1*BLK_K+k];
                register float a2 = a_buf[2*BLK_K+k], a3 = a_buf[3*BLK_K+k];
                register float b0 = b_buf[k*SUB_N+0],  b1 = b_buf[k*SUB_N+1];
                register float b2 = b_buf[k*SUB_N+2],  b3 = b_buf[k*SUB_N+3];
                asm volatile ("" ::: "memory");
                c00=fmaf(a0,b0,c00); c01=fmaf(a0,b1,c01);
                c02=fmaf(a0,b2,c02); c03=fmaf(a0,b3,c03);
                c10=fmaf(a1,b0,c10); c11=fmaf(a1,b1,c11);
                c12=fmaf(a1,b2,c12); c13=fmaf(a1,b3,c13);
                c20=fmaf(a2,b0,c20); c21=fmaf(a2,b1,c21);
                c22=fmaf(a2,b2,c22); c23=fmaf(a2,b3,c23);
                c30=fmaf(a3,b0,c30); c31=fmaf(a3,b1,c31);
                c32=fmaf(a3,b2,c32); c33=fmaf(a3,b3,c33);
            }
        }

        // Write 4x4 register block to DRAM
        register float tp00=c00, tp01=c01, tp02=c02, tp03=c03;
        register float tp10=c10, tp11=c11, tp12=c12, tp13=c13;
        register float tp20=c20, tp21=c21, tp22=c22, tp23=c23;
        register float tp30=c30, tp31=c31, tp32=c32, tp33=c33;
        asm volatile ("" ::: "memory");

        float *crow = &C[N * i_abs + j_abs];
        crow[0*N+0]=tp00; crow[0*N+1]=tp01; crow[0*N+2]=tp02; crow[0*N+3]=tp03;
        crow[1*N+0]=tp10; crow[1*N+1]=tp11; crow[1*N+2]=tp12; crow[1*N+3]=tp13;
        crow[2*N+0]=tp20; crow[2*N+1]=tp21; crow[2*N+2]=tp22; crow[2*N+3]=tp23;
        crow[3*N+0]=tp30; crow[3*N+1]=tp31; crow[3*N+2]=tp32; crow[3*N+3]=tp33;
    }

    // Remainder columns
    {
        int js_start = (my_N / SUB_N) * SUB_N;
        for (int ri = 0; ri < MICRO_M; ri++) {
            for (int mj = js_start; mj < my_N; mj++) {
                int j_abs = col_start + mj;
                register float acc = 0.0f;
                for (int kk = 0; kk < N; kk++)
                    acc = fmaf(A[N*(i_abs+ri)+kk], B[N*kk+j_abs], acc);
                C[N*(i_abs+ri)+j_abs] = acc;
            }
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

        // Micro-tiled row loop
        bsg_unroll(1)
        for (int ii = 0; ii < my_M; ii += MICRO_M) {
            int i_cnt = ((ii + MICRO_M) <= my_M) ? MICRO_M : (my_M - ii);
            int i_abs = row_start + ii;

            // Scalar fallback for remainder rows
            if (i_cnt < MICRO_M) {
                for (int ri = 0; ri < i_cnt; ri++)
                    for (int j = col_start; j < col_end; j++) {
                        register float acc = 0.0f;
                        for (int kk = 0; kk < N; kk++)
                            acc = fmaf(A[N*(i_abs+ri)+kk], B[N*kk+j], acc);
                        C[N*(i_abs+ri)+j] = acc;
                    }
                continue;
            }

            compute_c_tile(A, B, C, i_abs, col_start, my_N);
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

