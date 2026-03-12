# SGEMM Optimization on HammerBlade (BSG Manycore)

Progressive optimization of single-precision matrix multiply (SGEMM) targeting the HammerBlade BSG Manycore architecture with Vanilla-5 in-order RISC-V cores.

## Phase 0 — x86 Baseline

| File 					| Description 								|
|---------------------------------------|-----------------------------------------------------------------------|
| `sgemm_basic_x86.cpp` 		| Naive triple-loop SGEMM on x86 					|
| `sgemm_optimized_x86.cpp` 		| Tiled i-k-j with MICRO_M=4 micro-tiling, tuned for x86 L1 cache 	|
| `sgemm_common_x86.h` 			| Shared definitions 							|

## Phase 1 — Serial HB Kernels (Single Tile)

Only tile 0 computes; all other tiles idle at barriers. Each revision builds on the previous one.

| Rev | File			 	| Key Insight |
|-----|---------------------------------|-------------|
| 1.0 | `rev1_0_direct_port.cpp` 	| Direct port of x86 tiled i-k-j. Tile sizes (32/64/64) unchanged from x86. Establishes that the code runs correctly on HB but with high DRAM stall overhead. |
| 1.1 | `rev1_1_tile_tuning.cpp` 	| Reduced tile sizes to 16/32/32 to better fit HB's shared vcache hierarchy. Smaller working set reduces vcache thrashing. |
| 1.2 | `rev1_2_unroll_loadsep.cpp` 	| j-loop unrolled by 8 with load/compute/store phase separation. Groups non-blocking DRAM loads before FMAs to create load-use distance, giving the network time to return data. |
| 1.3 | `rev1_3_scratchpad.cpp` 	| A and C tiles copied into DMEM scratchpad. A reads become single-cycle local accesses instead of multi-cycle DRAM round-trips. B still read from DRAM. **Final serial baseline** — Phase 2 speedups are measured against this. |

## Phase 2 — Parallel HB Kernels (128 Cores, 16×8 Tile Group)

All 128 cores participate. Work is distributed across cores with barriers for synchronization.

| Rev | File 				| Key Insight |
|-----|---------------------------------|-------------|
| 2.0 | `rev2_0_row_parallel.cpp` | Cyclic row distribution — each core handles rows `i = tile_id, tile_id + 128, ...`. Simple but every core reads all N columns of B. |
| 2.1 | `rev2_1_2d_block.cpp` | 2D block decomposition (bsg_y=rows, bsg_x=cols). Each core only needs N/P_c columns of B, cutting per-core B traffic. A and C in DMEM; B still from DRAM. |
| 2.2 | `rev2_2_scratchpad_tiling.cpp` | B tile now also buffered in DMEM. All inner-loop reads (A, B, C) are local — zero network traffic during compute. |
| 2.3 | `rev2_3_load_pipeline.cpp` | DRAM→DMEM copies restructured as burst loads (8 outstanding requests before storing). Exploits non-blocking loads to hide network latency. |
| 2.4 | `rev2_4_double_buffer.cpp` | Double-buffered A/B — while computing on buf[cur], next k-tile loads into buf[nxt]. Overlaps load latency with compute. |


## Common Patterns

- **`bsg_fence()`** after C-zeroing ensures non-blocking DRAM stores complete before subsequent loads read them back.
- **Global DMEM buffers** (not stack-allocated) to avoid DMEM stack overflow on tile cores.
- **`asm volatile ("" ::: "memory")`** as a compiler barrier to separate load-issue from compute phases.
- **8-point `bsg_printf` checkpoints** in every revision for debugging execution flow.
