// Rev 2.0: Parallel SGEMM — cyclic row distribution across 128 cores.
// Each core accumulates its C row-strip in DMEM, then writes back to DRAM.

#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <cstring>
#include <cstdint>
#include <math.h>
#include "bsg_barrier_multipod.h"

volatile int done[NUM_POD_X]={0};
int alert = 0;

#define BLOCK_J 16

float c_buf[BLOCK_J];

extern "C"
int kernel(float *mat1, float *mat2, float *result, int pod_id)
{
    bsg_barrier_tile_group_init();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 1: tile_group_init + sync done\n");

    bsg_barrier_multipod(pod_id, NUM_POD_X, done, &alert);

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 2: multipod barrier done\n");

    bsg_cuda_print_stat_kernel_start();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 3: stat_kernel_start done\n");

    int num_tiles = bsg_tiles_X * bsg_tiles_Y;
    int tile_id   = __bsg_id;

    if (__bsg_id == 0) bsg_printf("[DBG] num_tiles=%d, N=%d, NITER=%d\n", num_tiles, N, NITER);

    bsg_unroll(1)
    for (int iter = 0; iter < NITER; iter++) {
        float *A = &mat1[N * N * iter];
        float *B = &mat2[N * N * iter];
        float *C = &result[N * N * iter];

        // Cyclic row distribution
        bsg_unroll(1)
        for (int i = tile_id; i < N; i += num_tiles) {

            bsg_unroll(1)
            for (int jj = 0; jj < N; jj += BLOCK_J) {

                // Zero local accumulator
                bsg_unroll(BLOCK_J)
                for (int j = 0; j < BLOCK_J; j++) {
                    c_buf[j] = 0.0f;
                }

                // Accumulate C row-strip
                bsg_unroll(1)
                for (int k = 0; k < N; k++) {
                    float a_val = A[N * i + k];

                    bsg_unroll(BLOCK_J)
                    for (int j = 0; j < BLOCK_J; j++) {
                        c_buf[j] = fmaf(a_val, B[N * k + jj + j], c_buf[j]);
                    }
                }

                // Write back to DRAM
                bsg_unroll(BLOCK_J)
                for (int j = 0; j < BLOCK_J; j++) {
                    C[N * i + jj + j] = c_buf[j];
                }
            }

            if (__bsg_id == 0) bsg_printf("[DBG] tile0: iter=%d row=%d done\n", iter, i);
        }

        if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 4: iter %d complete\n", iter);
    }

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 5: all computation done\n");

    bsg_fence();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 6: fence1 done\n");

    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 7: barrier1 done\n");

    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    bsg_barrier_tile_group_sync();

    if (__bsg_id == 0) bsg_printf("[DBG] checkpoint 8: returning\n");

    return 0;
}

