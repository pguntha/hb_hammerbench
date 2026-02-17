# Makefile for x86 SGEMM Benchmarking
# ====================================
#
# Builds both basic (naive) and optimized SGEMM kernels with:
#   - Core pinning (compile-time + runtime via taskset)
#   - Parameterized tile sizes for the optimized version
#   - Configurable matrix dimensions and iteration counts
#
# Usage:
#   make                    # Build both binaries with defaults
#   make run                # Build + run both on pinned core
#   make run-basic          # Build + run basic only
#   make run-opt            # Build + run optimized only
#   make sweep              # Run optimized with multiple tile configs
#   make clean              # Remove binaries
#
# Override parameters on the command line:
#   make run M=1024 K=1024 N=1024
#   make run-opt TILE_M=16 TILE_K=64 TILE_N=64 MICRO_M=4
#   make run PIN_CORE=2 NUM_ITERS=11

# ── Compiler settings ──
CXX      = g++
CXXFLAGS_COMMON = -Wall -Wextra -lm

# Basic: -O2 only (no aggressive opts -- the point is to be naive)
CXXFLAGS_BASIC = -O2 $(CXXFLAGS_COMMON)

# Optimized: full optimization + native ISA (SSE4.2 on X5650, no AVX)
CXXFLAGS_OPT   = -O3 -march=native -funroll-loops $(CXXFLAGS_COMMON)

# ── Core pinning ──
# Logical core to pin to (physical core 1, socket 0 on X5650)
# Core 1 is preferred over core 0 to avoid OS housekeeping interference.
PIN_CORE ?= 1

# ── Benchmark parameters ──
NUM_ITERS ?= 7
M         ?= 512
K         ?= 512
N         ?= 512

# ── Tile parameters for optimized version ──
# Defaults tuned for Xeon X5650 (32 KB L1d):
#   A tile: 32×32×4 = 4 KB
#   B tile: 32×64×4 = 8 KB
#   C tile: 32×64×4 = 8 KB
#   Total:  20 KB (~63% of L1d, leaves headroom)
TILE_M  ?= 32
TILE_K  ?= 32
TILE_N  ?= 64
MICRO_M ?= 4

# ── Compile-time defines ──
DEFINES_BASIC = -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE)
DEFINES_OPT   = -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
                -DTILE_M=$(TILE_M) -DTILE_K=$(TILE_K) \
                -DTILE_N=$(TILE_N) -DMICRO_M=$(MICRO_M)

# ── Targets ──
BASIC_BIN = sgemm_basic
OPT_BIN   = sgemm_opt

.PHONY: all clean run run-basic run-opt sweep help

all: $(BASIC_BIN) $(OPT_BIN)

$(BASIC_BIN): sgemm_basic_x86.cpp sgemm_common_x86.h
	$(CXX) $(CXXFLAGS_BASIC) $(DEFINES_BASIC) -o $@ $<

$(OPT_BIN): sgemm_optimized_x86.cpp sgemm_common_x86.h
	$(CXX) $(CXXFLAGS_OPT) $(DEFINES_OPT) -o $@ $<

clean:
	rm -f $(BASIC_BIN) $(OPT_BIN)

# ── Run targets ──
# taskset provides runtime pinning as a safety net on top of the
# compile-time sched_setaffinity() call.  Belt and suspenders.

run: all
	@echo ""
	@echo "================================================================"
	@echo " X86 SGEMM BENCHMARK"
	@echo " Matrix: $(M) x $(K) x $(N)"
	@echo " Pinned to core: $(PIN_CORE)"
	@echo " Iterations: $(NUM_ITERS)"
	@echo "================================================================"
	@echo ""
	@echo "────────────────────── BASIC (Naive) ──────────────────────"
	taskset -c $(PIN_CORE) ./$(BASIC_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "────────────────────── OPTIMIZED ──────────────────────────"
	@echo " Tiles: M=$(TILE_M) K=$(TILE_K) N=$(TILE_N) MICRO_M=$(MICRO_M)"
	taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "================================================================"

run-basic: $(BASIC_BIN)
	taskset -c $(PIN_CORE) ./$(BASIC_BIN) $(M) $(K) $(N)

run-opt: $(OPT_BIN)
	taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)

# ── Tile sweep ──
# Tries several tile configurations to find the best for this CPU.
# Each config rebuilds the optimized binary with different -D flags.
sweep: $(BASIC_BIN)
	@echo ""
	@echo "================================================================"
	@echo " TILE SIZE SWEEP -- Matrix: $(M) x $(K) x $(N), Core: $(PIN_CORE)"
	@echo "================================================================"
	@echo ""
	@echo "── Baseline: Basic (Naive) ──"
	@taskset -c $(PIN_CORE) ./$(BASIC_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 1: TILE_M=32 TILE_K=32 TILE_N=64 MICRO_M=4 (20 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=32 -DTILE_K=32 -DTILE_N=64 -DMICRO_M=4 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 2: TILE_M=16 TILE_K=64 TILE_N=64 MICRO_M=4 (24 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=16 -DTILE_K=64 -DTILE_N=64 -DMICRO_M=4 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 3: TILE_M=32 TILE_K=64 TILE_N=64 MICRO_M=4 (32 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=32 -DTILE_K=64 -DTILE_N=64 -DMICRO_M=4 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 4: TILE_M=64 TILE_K=32 TILE_N=32 MICRO_M=4 (16 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=64 -DTILE_K=32 -DTILE_N=32 -DMICRO_M=4 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 5: TILE_M=16 TILE_K=32 TILE_N=128 MICRO_M=4 (24 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=16 -DTILE_K=32 -DTILE_N=128 -DMICRO_M=4 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "── Config 6: TILE_M=32 TILE_K=32 TILE_N=64 MICRO_M=8 (20 KB) ──"
	$(CXX) $(CXXFLAGS_OPT) -DNUM_ITERS=$(NUM_ITERS) -DPIN_CORE=$(PIN_CORE) \
		-DTILE_M=32 -DTILE_K=32 -DTILE_N=64 -DMICRO_M=8 \
		-o $(OPT_BIN) sgemm_optimized_x86.cpp
	@taskset -c $(PIN_CORE) ./$(OPT_BIN) $(M) $(K) $(N)
	@echo ""
	@echo "================================================================"
	@echo " SWEEP COMPLETE -- compare median GFLOPS above to pick winner"
	@echo "================================================================"

help:
	@echo "x86 SGEMM Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        Build both basic and optimized binaries (default)"
	@echo "  run        Build + run both, pinned to core $(PIN_CORE)"
	@echo "  run-basic  Build + run basic only"
	@echo "  run-opt    Build + run optimized only"
	@echo "  sweep      Try multiple tile configs, compare results"
	@echo "  clean      Remove binaries"
	@echo "  help       This message"
	@echo ""
	@echo "Parameters (override on command line):"
	@echo "  M=512 K=512 N=512    Matrix dimensions"
	@echo "  PIN_CORE=1           Logical core to pin to"
	@echo "  NUM_ITERS=7          Benchmark iterations"
	@echo "  TILE_M=32            Tile height (rows of A/C)"
	@echo "  TILE_K=32            Tile depth (cols of A / rows of B)"
	@echo "  TILE_N=64            Tile width (cols of B/C)"
	@echo "  MICRO_M=4            Register micro-tile rows"
	@echo ""
	@echo "Examples:"
	@echo "  make run M=1024 K=1024 N=1024"
	@echo "  make run-opt TILE_M=16 TILE_K=64 TILE_N=64"
	@echo "  make sweep PIN_CORE=2 M=256 K=256 N=256"

