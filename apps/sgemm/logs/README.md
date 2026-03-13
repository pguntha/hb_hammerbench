# SGEMM Execution Results on HammerBlade

All execution results in this repository correspond to **64×64 single-precision (IEEE 754 `float`, 32-bit) matrix multiplication**, i.e., SGEMM: `C = A × B`, where `A`, `B`, and `C` are each 64×64 matrices of type `float`.

The matrix dimension `N=64` is supplied as a compile-time define (`-DN=64`) and is shared by both the host-side driver and the RISC-V device kernel. Each run executes `NITER=2` independent multiplications per invocation (the default value), operating on two distinct matrix pairs in sequence.

## Matrix Initialization

Both input matrices use a deterministic, reproducible initialization pattern:

```c
mat1[i] = (float)(i % 7);
mat2[i] = (float)(i % 3);
```

This applies element-by-element across the flattened `NITER × N × N` buffer. The pattern avoids special values (zeros, ones, powers-of-two) that could mask numerical bugs, while remaining fully deterministic across runs.

## Correctness Verification

Correctness is verified by the host program (`main.cpp`) using a two-step process:

1. **Golden reference computation**: The host computes the expected result using a naive triple-nested loop in single-precision (`float`) arithmetic:
   ```c
   for (int y = 0; y < N; y++)
     for (int x = 0; x < N; x++) {
       float sum = 0.0f;
       for (int z = 0; z < N; z++)
         sum += mat1[(N*y)+z] * mat2[(N*z)+x];
       result[(N*y)+x] = sum;
     }
   ```

2. **Element-wise Sum of Squared Errors (SSE)**: After the device kernel completes and results are DMA-transferred back to the host, every element of the device output is compared against the golden reference. The SSE is accumulated as:
   ```c
   float diff = expected - actual;
   sse += diff * diff;
   ```
   Any element with a nonzero difference is individually logged. The run **passes** if the total SSE across all `NITER × N × N` elements is below `0.01`; otherwise it is flagged as a mismatch.

Because both the host reference and the device kernel operate in single-precision (`float`) arithmetic on the same input data, exact bitwise agreement (SSE = 0.0) is expected for correct kernels at `N=64`. The `0.01` threshold provides a margin for potential floating-point reordering in more aggressively optimized revisions.

## Common Configuration Across All Kernels

The following settings were held constant for every execution result file in this repository:

| Parameter | Value | Notes |
|---|---|---|
| Matrix dimension (`N`) | 64 | Square matrices: `A[64×64] × B[64×64] = C[64×64]` |
| Datatype | `float` (32-bit) | IEEE 754 single-precision |
| Iterations (`NITER`) | 2 | Two independent multiplications per run |
| Grid dimension | 1×1 | Single tile group launched |
| Data transfer | DMA | Host-to-device (`htod`) and device-to-host (`dtoh`) |
| Profiling region | `bsg_cuda_print_stat_kernel_start()` to `bsg_cuda_print_stat_kernel_end()` | Only the kernel compute phase is profiled; DMA and host-side setup are excluded |
| Multipod barrier | `bsg_barrier_multipod()` | Called before the profiling start marker to synchronize all pods |
| Kernel interface | `kernel(float *mat1, float *mat2, float *result, int pod_id)` | Uniform across all revisions |

### Phase-Specific Tile Group Configuration

| Phase | Revisions | Tile Group | Cores |
|---|---|---|---|
| Phase 1 (Serial) | `rev1_0` through `rev1_3` | 1×1 | 1 core |
| Phase 2 (Parallel) | `rev2_0` through `rev2_8` | 16×8 | 128 cores |

Phase 1 kernels run on a single core to establish a serial baseline. Phase 2 kernels distribute work across all 128 cores in a 16×8 tile configuration (the full HammerBlade pod).

## Execution Result Format

Each `results_<revision>.txt` file contains the profiler output with three sections:

- **DRAM Utilization** - Read/write/busy/idle percentages for off-chip memory
- **Vcache Utilization** - Hit/miss rates, load/store/idle breakdown for the shared victim cache (32 banks)
- **Core Utilization** - Cycle-level breakdown including stall categories (DRAM load dependency, barrier, fence, remote request), bubble cycles (branch mispredictions), and a full instruction mix with per-opcode counts and percentages. Ends with overall utilization percentage and total runtime in cycles.
