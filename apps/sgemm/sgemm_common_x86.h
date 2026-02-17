// sgemm_common_x86.h
// Shared utilities for x86 SGEMM benchmarks: timing, matrix init,
// correctness verification, and benchmark harness.

#ifndef SGEMM_COMMON_X86_H
#define SGEMM_COMMON_X86_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>

// For core pinning (useful to run the benchmarks on a specific core - reliable comparision)
#ifdef __linux__
#include <sched.h>
#endif

// Default benchmark iterations (override with -DNUM_ITERS=N at compile time)
#ifndef NUM_ITERS
#define NUM_ITERS 7
#endif

// Wall-clock timer with nanosecond resolution
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Pin this process to a specific logical CPU core
// Returns 0 on success, -1 on failure.  No-op on non-Linux.
static int pin_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rc == 0) {
        printf("[affinity] Pinned to logical core %d\n", core_id);
    } else {
        perror("[affinity] sched_setaffinity failed");
    }
    return rc;
#else
    (void)core_id;
    printf("[affinity] Core pinning not supported on this OS\n");
    return -1;
#endif
}

// Fill a matrix with deterministic pseudo-random values in [-1, 1]
// Same seed = same data, so basic and optimized versions can cross-check.
static void init_matrix(float* M, int rows, int cols, int seed) {
    for (int i = 0; i < rows * cols; i++) {
        M[i] = ((float)((i * seed + 7) % 1000)) / 500.0f - 1.0f;
    }
}

// Spot-check C entries against a naive recomputation
// Returns the largest absolute error found.
static float verify(const float* A, const float* B, const float* C,
                    int M, int K, int N, int num_checks) {
    float max_err = 0.0f;
    srand(42);
    for (int check = 0; check < num_checks; check++) {
        int i = rand() % M;
        int j = rand() % N;
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += A[i * K + k] * B[k * N + j];
        }
        float err = fabsf(C[i * N + j] - expected);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Allocate three matrices and fill A, B with deterministic data
// Returns false on allocation failure.
static bool alloc_and_init(float** A, float** B, float** C,
                           int M, int K, int N) {
    *A = (float*)malloc(M * K * sizeof(float));
    *B = (float*)malloc(K * N * sizeof(float));
    *C = (float*)malloc(M * N * sizeof(float));
    if (!*A || !*B || !*C) {
        fprintf(stderr, "Memory allocation failed\n");
        return false;
    }
    init_matrix(*A, M, K, 13);
    init_matrix(*B, K, N, 29);
    return true;
}

// Parse optional [M K N] from command line; defaults to 512x512x512
static void parse_dims(int argc, char** argv, int& M, int& K, int& N) {
    M = 512; K = 512; N = 512;
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
}

// Statistical benchmark harness
// Runs kernel NUM_ITERS times, collects all timings, sorts them,
// and reports min / median / max / stddev.  Returns the median time.
template<typename Func>
static double benchmark(Func kernel,
                        const float* A, const float* B, float* C,
                        int M, int K, int N, int num_iters = NUM_ITERS) {
    double* times = new double[num_iters];

    for (int iter = 0; iter < num_iters; iter++) {
        double t0 = get_time_sec();
        kernel(A, B, C, M, K, N);
        double t1 = get_time_sec();
        times[iter] = t1 - t0;
        printf("  Iteration %d: %.6f sec\n", iter, times[iter]);
    }

    // Sort for median / min / max
    std::sort(times, times + num_iters);

    double min_t = times[0];
    double max_t = times[num_iters - 1];
    double median_t;
    if (num_iters % 2 == 1) {
        median_t = times[num_iters / 2];
    } else {
        median_t = (times[num_iters / 2 - 1] + times[num_iters / 2]) / 2.0;
    }

    // Compute mean and stddev
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < num_iters; i++) {
        sum += times[i];
        sum_sq += times[i] * times[i];
    }
    double mean = sum / num_iters;
    double variance = (sum_sq / num_iters) - (mean * mean);
    double stddev = (variance > 0.0) ? sqrt(variance) : 0.0;

    printf("\n  Timing stats (%d iterations):\n", num_iters);
    printf("    Min:    %.6f sec\n", min_t);
    printf("    Median: %.6f sec\n", median_t);
    printf("    Max:    %.6f sec\n", max_t);
    printf("    Mean:   %.6f sec\n", mean);
    printf("    StdDev: %.6f sec  (%.2f%%)\n", stddev, (stddev / mean) * 100.0);

    delete[] times;
    return median_t;
}

// Print GFLOPS and run spot-check verification
// Uses the provided time (typically the median) for GFLOPS calculation.
static void report_and_verify(double time_sec,
                              const float* A, const float* B, const float* C,
                              int M, int K, int N) {
    double flops = 2.0 * M * N * K;
    double gflops = flops / time_sec / 1e9;
    printf("\nMedian time: %.6f sec\n", time_sec);
    printf("Performance: %.2f GFLOPS\n", gflops);

    float max_err = verify(A, B, C, M, K, N, 100);
    printf("Max error (100 spot checks): %e\n", max_err);
    printf("%s\n", (max_err < 1e-3f) ? "PASS" : "FAIL -- error too large");
}

#endif // SGEMM_COMMON_X86_H
