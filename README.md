# PTMS — Parallel Tournament Merge Sort

[![CI](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml/badge.svg)](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A parallel merge sort for C++17 with OpenMP and AVX2, plus a
template-friendly generic pipeline for arbitrary types.

**PTMS is not a new algorithm.** It is a carefully optimised
implementation of classic 2-way tournament merge sort, composed of known
techniques: parallel chunking, tournament (multi-way) merging at the top
levels, AVX2-accelerated base cases, and a small set of sorting networks
for tiny inputs. Nothing here is novel in a research sense. What is here
is a fast, portable, type-generic merge sort that is competitive with
`tbb::parallel_sort` and with parallel radix sort on 64-bit keys, and
that **outperforms IPS4o on cache-resident inputs** (see
[Benchmarks](#benchmarks)).

---

## Contents

- [What it is](#what-it-is)
- [Architecture](#architecture)
- [API](#api)
- [Building](#building)
- [Benchmarks](#benchmarks)
- [Architectural limitations](#architectural-limitations)
- [Summary](#summary)
- [License](#license)

---

## What it is

PTMS sorts arrays of `int32` and arrays of arbitrary trivially-copyable
types. It runs on top of OpenMP, uses AVX2 for the `int32` fast path
(AVX-512 code paths exist but are not exercised on current test
hardware), and exposes a small set of knobs for thread control.

Target hardware for the reference numbers below:

- AMD Ryzen 9 6900HX (8C/16T, 16 MB L3, AVX2, no AVX-512)
- Windows 11, clang-cl (Visual Studio 2022 toolchain)
- TBB 2023.1.0 via NuGet (`inteltbb.devel.win` + `inteltbb.redist.win`)

---

## Architecture

Three-level pipeline:

1. **Top level — tournament merge.** P independent sorted runs are
   merged pairwise in a tournament tree (log₂ P levels), using AVX2
   kernels where the element type allows. This keeps the number of
   full-memory passes close to `log₂ P` rather than `log₂ N`.
2. **Middle level — parallel chunk sort.** The input is split into
   chunks; each chunk is sorted in parallel (one OpenMP task per chunk
   with `#pragma omp taskgroup`). Chunk count and thread count are
   chosen by `ThreadMode` (see [API](#api)).
3. **Base level — small cases.** Runs of size 4..7 use hand-written
   sorting networks. Runs that fit a single AVX2 register use
   vectorised merge kernels. Everything below the base case is
   scalar-branchless.

Data layout: caller provides an auxiliary buffer of the same length as
the input; PTMS does not allocate on the hot path. All sorts are
out-of-place at the top level and in-place at the base levels.

---

## API

```cpp
#include "ptms.h"

// Fast path: int32, AVX2-accelerated.
void ptms::sort_int(int* a, int* buf, int n,
                    ptms::ThreadMode mode = ptms::ThreadMode::Auto,
                    int fixed_threads = 0);

// Batch: many arrays of the same size; reuses one OpenMP region.
void ptms::sort_int_batch(int* data, int* buf, int n, int num_arrays,
                          ptms::ThreadMode mode = ptms::ThreadMode::Auto,
                          int fixed_threads = 0);

// Generic templated pipeline for arbitrary trivially-copyable types.
// Specialised for int -> routes to sort_int.
template <class T>
void ptms::sort(T* a, T* buf, int n,
                ptms::ThreadMode mode = ptms::ThreadMode::Auto,
                int fixed_threads = 0);

// Runtime introspection.
const char* ptms::simd_level();          // "AVX-512" | "AVX2" | "scalar"
int         ptms::physical_core_count(); // excludes SMT siblings
```

### `ThreadMode`

| Mode | Meaning |
|---|---|
| `Auto` | Choose thread count from array size: 1 / 2 / 4 / 8 / all, with thresholds at N = 512 / 2048 / 8192 / 32768. |
| `Single` | Always single-threaded. Useful for benchmarking and for tiny arrays. |
| `ByCores` | Use physical cores only (no SMT). |
| `Fixed` | Use exactly `fixed_threads` threads. |

The `Auto` thresholds were tuned on the reference machine; a different
NUMA topology or core count may shift the sweet spots by one step.

---

## Building

Requirements:

- C++17 compiler. Tested with clang-cl (VS 2022), GCC, MinGW-w64, Apple
  Clang.
- OpenMP. On MSVC-family toolchains, use clang-cl — MSVC proper does
  not ship `taskgroup` support.
- CMake ≥ 3.15.
- (Optional) Intel TBB for the benchmark's `tbb::parallel_sort`
  comparison. On Windows, the CMakeLists auto-detects TBB installed via
  the Visual Studio NuGet packages `inteltbb.devel.win` and
  `inteltbb.redist.win`.

```bash
git clone https://github.com/TylerBurger777/PTMS.git
cd PTMS
cmake -S . -B out/build/x64-Release -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/x64-Release --config Release
```

Build artefacts:

- `benchmark` — PTMS vs `tbb::parallel_sort` vs `std::sort`, five
  element types, logarithmic N progression, interleaved rounds.
- `test_correctness` — ~130 correctness cases: powers of two, 2^k ± 1,
  n < 4, odd thread counts, patterns (already sorted, reverse, all
  equal, single outlier), batch API.

### SIMD selection

The CMake option `SIMD` selects the vectorisation target:

- `AUTO` (default) uses `-march=native`. On MSVC-family toolchains
  where `-march=native` is unavailable, it falls back to `/arch:AVX2`.
- `AVX512`, `AVX2`, `SCALAR` force a specific level.

Use `SCALAR` for cross-compiled binaries or when targeting CPUs without
AVX2.

---

## Benchmarks

Methodology:

- Interleaved rounds (each algorithm is timed once per round, in a
  fixed order, over multiple rounds) to cancel out thermal drift.
- Median of runs after MAD-outlier filtering (threshold 3.5).
- 1000 ms cooldown between timed runs. Without cooldown, jitter on a
  laptop reaches 60%; with it, 2–15%.
- Batch size 5, `std::sort` skipped above N = 8 388 608.

### PTMS vs `tbb::parallel_sort`, N = 4M

| Type | PTMS | TBB | Ratio |
|---|---|---|---|
| int32 | 17.9 ms | 34.5 ms | **1.93x** |
| int64 | 35.0 ms | 38.0 ms | 1.09x |
| float | 25.9 ms | 47.8 ms | **1.84x** |
| double | 40.0 ms | 48.8 ms | 1.22x |
| string (N = 1M) | 137 ms | 89 ms | 0.65x |

### PTMS vs parallel radix sort, 64-bit keys

| Type | N | PTMS | radix | Ratio |
|---|---|---|---|---|
| int64 | 2M | 15.0 | 17.2 | 1.15x |
| int64 | 4M | 35.0 | 42.3 | 1.21x |
| int64 | 8M | 75.5 | 84.4 | 1.12x |
| double | 2M | 16.9 | 20.5 | 1.21x |
| double | 4M | 40.0 | 47.3 | 1.18x |
| double | 8M | 84.7 | 94.5 | 1.11x |
| double | 16M | 178.6 | 191.9 | 1.07x |

Why PTMS wins against LSD radix on 64-bit keys: radix does
`sizeof(T)` passes over `sizeof(T)` bytes, i.e. cost scales as
`sizeof(T)²`. PTMS does `log₂(N)` merge levels over `sizeof(T)` bytes,
i.e. cost scales as `sizeof(T)`. Going from int32 to int64 makes radix
4x more expensive but PTMS only 2x.

### PTMS vs IPS4o across sizes — the crossover

IPS4o is the state-of-the-art parallel samplesort and, on large inputs,
is the fastest sorter in the comparison set. But IPS4o pays a
**fixed setup cost**: it extracts and sorts `kBuckets × kOversample =
16384` sample elements on every call, computes 255 splitters, and
initialises a 256-bucket classifier. On cache-resident inputs, this
setup dominates and PTMS wins.

The table below is from a same-machine run (`bench_vs_ips4o`, adaptive
cooldown 200/500/1000 ms by N, runs = 5, warmup = 2, median). Ratio
< 1.0 means PTMS is faster; > 1.0 means IPS4o is faster.

**int32:**

| N | PTMS | IPS4o | Ratio | Winner |
|---|---|---|---|---|
| 10K | 0.10 ms | 0.38 ms | 0.28 | **PTMS 3.6x** |
| 20K | 0.18 ms | 0.51 ms | 0.35 | **PTMS 2.9x** |
| 40K | 0.32 ms | 0.97 ms | 0.33 | **PTMS 3.0x** |
| 80K | 0.54 ms | 1.94 ms | 0.28 | **PTMS 3.6x** |
| 160K | 1.08 ms | 1.56 ms | 0.69 | **PTMS 1.4x** |
| 320K | 2.06 ms | 3.53 ms | 0.58 | **PTMS 1.7x** |
| 640K | 2.50 ms | 3.06 ms | 0.82 | **PTMS 1.2x** |
| 1.28M | 4.59 ms | 5.18 ms | 0.89 | **PTMS 1.1x** |
| 2.56M | 9.64 ms | 7.50 ms | 1.29 | IPS4o 1.3x |
| 4M | 20.4 ms | 10.2 ms | 2.00 | IPS4o 2.0x |

**int64:**

| N | PTMS | IPS4o | Ratio | Winner |
|---|---|---|---|---|
| 10K | 0.12 ms | 0.38 ms | 0.32 | **PTMS 3.2x** |
| 20K | 0.20 ms | 1.15 ms | 0.17 | **PTMS 5.9x** |
| 40K | 0.32 ms | 4.73 ms | 0.07 | **PTMS 14.9x** (see note) |
| 80K | 0.67 ms | 1.42 ms | 0.47 | **PTMS 2.1x** |
| 160K | 1.09 ms | 1.80 ms | 0.60 | **PTMS 1.7x** |
| 320K | 2.12 ms | 1.95 ms | 1.09 | IPS4o 1.1x |
| 640K | 3.95 ms | 3.18 ms | 1.24 | IPS4o 1.2x |
| 1.28M | 5.87 ms | 4.69 ms | 1.25 | IPS4o 1.3x |
| 2.56M | 18.6 ms | 9.57 ms | 1.94 | IPS4o 1.9x |
| 4M | 32.1 ms | 14.6 ms | 2.20 | IPS4o 2.2x |

**float:**

| N | PTMS | IPS4o | Ratio | Winner |
|---|---|---|---|---|
| 10K | 0.14 ms | 0.37 ms | 0.37 | **PTMS 2.7x** |
| 20K | 0.23 ms | 0.67 ms | 0.34 | **PTMS 2.9x** |
| 40K | 0.42 ms | 32.4 ms | 0.01 | **PTMS 78x** (see note) |
| 80K | 0.76 ms | 5.00 ms | 0.15 | **PTMS 6.6x** |
| 160K | 1.52 ms | 1.96 ms | 0.78 | **PTMS 1.3x** |
| 320K | 2.03 ms | 1.90 ms | 1.07 | IPS4o 1.1x |
| 640K | 4.70 ms | 3.09 ms | 1.52 | IPS4o 1.5x |
| 1.28M | 6.51 ms | 4.99 ms | 1.30 | IPS4o 1.3x |
| 2.56M | 13.3 ms | 7.85 ms | 1.70 | IPS4o 1.7x |
| 4M | 23.8 ms | 12.1 ms | 1.96 | IPS4o 2.0x |

**double:**

| N | PTMS | IPS4o | Ratio | Winner |
|---|---|---|---|---|
| 10K | 0.15 ms | 0.43 ms | 0.35 | **PTMS 2.9x** |
| 20K | 0.25 ms | 1.23 ms | 0.21 | **PTMS 4.8x** |
| 40K | 0.38 ms | 3.59 ms | 0.11 | **PTMS 9.4x** (see note) |
| 80K | 0.82 ms | 1.40 ms | 0.59 | **PTMS 1.7x** |
| 160K | 1.52 ms | 4.27 ms | 0.36 | **PTMS 2.8x** |
| 320K | 2.04 ms | 5.53 ms | 0.37 | **PTMS 2.7x** |
| 640K | 5.01 ms | 3.76 ms | 1.33 | IPS4o 1.3x |
| 1.28M | 7.13 ms | 4.94 ms | 1.44 | IPS4o 1.4x |
| 2.56M | 29.1 ms | 9.52 ms | 3.05 | IPS4o 3.1x |
| 4M | 36.3 ms | 15.8 ms | 2.31 | IPS4o 2.3x |

**Note on outlier rows.** Some IPS4o measurements at N = 40K
(32.4 ms for float, 4.73 ms for int64, 3.59 ms for double) are 5–80x
higher than the trend on either side of N = 40K. This is not an
algorithmic effect of PTMS; it is an IPS4o-side anomaly on this
particular size. The exact cause is not diagnosed. PTMS is unaffected.
The rows are reported as measured.

### Crossover points

| Type | Size of element | Crossover N | Working set at crossover |
|---|---|---|---|
| int32 | 4 B | **2.56M** | 10.2 MB |
| int64 | 8 B | **320K** | 2.5 MB |
| float | 4 B | **320K** | 1.25 MB |
| double | 8 B | **640K** | 5 MB |

Two observations:

1. **PTMS wins below the crossover**, by 1.1–10x depending on type and
   size. On cache-resident data, IPS4o's fixed setup cost and its
   bucket-classification overhead dominate; PTMS's register-to-register
   merge path does not have these costs.
2. **int32 has the highest crossover (2.56M elements, ~10 MB)**,
   because PTMS has an AVX2 fast path only for `int32`. For int64,
   float, and double the generic pipeline runs without SIMD kernels,
   and IPS4o overtakes PTMS earlier (0.32M–0.64M elements, 1.25–5 MB).

Above the crossover, IPS4o's `log₂₅₆(N)` memory passes beat PTMS's
`log₂(N)` passes, and the ratio grows to ~2x at N = 4M. See
[Architectural limitations](#architectural-limitations) for the model.

---

## Architectural limitations

**PTMS is not a state-of-the-art parallel sorter for large inputs.** It
is a well-optimised 2-way merge sort with a genuine advantage on
cache-resident data. Above the crossover points listed above, where the
working set no longer fits in L3, IPS4o wins and the gap grows. The
following limitations are measurable facts.

### 1. It loses to IPS4o on DRAM-bound inputs

At N ≥ 4M int32 and N ≥ 640K for 8-byte types, PTMS is slower than
IPS4o. The gap grows with N:

| Type | N | PTMS | IPS4o | PTMS / IPS4o |
|---|---|---|---|---|
| int32 | 4M | 20.4 ms | 10.2 ms | 2.00x |
| int32 | 16M | 94.3 ms | 66.4 ms | 1.42x |
| int64 | 4M | 32.1 ms | 14.6 ms | 2.20x |
| int64 | 16M | 159.5 ms | 56.9 ms | 2.80x |
| double | 4M | 36.3 ms | 15.8 ms | 2.31x |
| double | 16M | 164.7 ms | 84.2 ms | 1.96x |

**Why.** IPS4o is a partition-based sorter: it does
`log₂₅₆(N) ≈ log₂(N)/8` memory passes. PTMS is a merge-based sorter:
it does `log₂(N)` memory passes. On DRAM-bound data, memory traffic is
the limiting factor, and the ratio of passes (`log₂(N) /
log₂(N)/8 = 8`) is partially offset by IPS4o's higher per-element CPU
cost, giving the ~2x measured gap.

### 2. Multi-way experiments did not close the gap

The following were implemented and benchmarked locally (not in this
repository):

- **Heap-based k-way merge** with k = 4 / 16 / 64 / 256. Result:
  170 / 165 / 215 / 250 ms at N = 4M int32, versus PTMS at 17.1 ms.
  10–15x slower than PTMS. Root cause: log₂(k) random heap accesses per
  element.
- **Sample sort (single-level, with staging).** Best result 33.1 ms at
  k = 256, cap = 128 — still ~2x slower than PTMS on the same input.
- **Single-level bucket sort** with direct bucket index computation
  (`(key ^ 0x80000000) >> (32 - log₂ k)`), no `lower_bound`. Best
  result 16.4 ms at k = 256, cap = 128 — actually **4.5% faster than
  PTMS** on N = 4M int32. But 1.6x slower than parallel radix and 1.6x
  slower than IPS4o. This is a distributing sort, not a multi-way
  samplesort.
- **PTMS8 — 8-way tournament merge**, with an 8-way loser tree both at
  the top level and inside chunks. Tested at N = 1M–16M across four
  types. Result: 3–10x **slower** than 2-way PTMS on the reference
  hardware. Root causes: (a) scalar loser tree has 3 dependent
  comparisons per output element and no SIMD; (b) OpenMP overhead on
  thousands of small base-case calls; (c) on cache-resident data the
  register-to-register 2-way bitonic merge is 4x faster than any
  memory-based k-way merge, regardless of k. 8-way merge is not a win
  below the L3 threshold.

Conclusion: naive multi-way loses to 2-way merge on cache-resident
data, and a proper samplesort (IPS4o) wins above the L3 threshold.
There is no configuration of merge sort that beats IPS4o on uniform
random data at DRAM-bound sizes; the algorithmic constant factors
favour partition-based sorters there.

### 3. `std::string` is slow

PTMS sorts `std::string` ~1.7x slower than `tbb::parallel_sort`. The
generic pipeline does not have a specialised kernel for non-trivial
types and falls back to moving whole `std::string` objects (32 bytes
each on libstdc++ / libc++), with no small-string-optimisation-aware
handling. Users who need fast string sorting should not use PTMS.

### 4. Almost-sorted inputs are a weak spot

On inputs with long sorted runs, `tbb::parallel_sort` outperforms PTMS
by up to 2x. PTMS does have an `is_sorted` fast path in both
`recursiveSortPP` and `middleAlgorithm`, but it does not detect
long-runs structure the way natural merge sort would. A natural-merge
extension is on the "maybe" list; it is not implemented.

### 5. AVX-512 code paths are untested

AVX-512 kernels exist in the source but there is no AVX-512 hardware in
the test matrix. On the Ryzen 9 6900HX, `simd_level()` correctly
reports `AVX2`. Do not assume the AVX-512 paths are correct.

### 6. Runtime `simd_level()` reflects the CPU, not the build

`simd_level()` uses `__builtin_cpu_supports` and reports what the CPU
supports, not which code paths were compiled in. If the library was
built with `-DSIMD=SCALAR`, `simd_level()` will still return `AVX2` on
an AVX2 machine, because it is describing the machine. To know which
kernels are active, inspect the build configuration at compile time.

---

## Summary

**Use PTMS when** your input fits in L3 cache. Concretely, for the
reference hardware (16 MB L3):

- `int32`, N ≤ ~2.5M (10 MB)
- `int64`, `double`, N ≤ ~320K–640K (2.5–5 MB)
- `float`, N ≤ ~320K (1.25 MB)

In this range PTMS is 1.1–10x faster than IPS4o and 1.1–1.9x faster
than `tbb::parallel_sort` depending on type.

PTMS also has non-performance advantages regardless of size:

- **No TBB dependency.** OpenMP only. Useful in restricted build
  environments where pulling in oneTBB is not viable.
- **Template-generic API.** `ptms::sort<T>` works for any
  trivially-copyable type without per-type kernels.
- **Small code base.** ~1500 lines versus IPS4o's ~15K. Easier to
  audit and embed.

**Do not use PTMS when:**

- Your input exceeds the L3 threshold. **Use IPS4o.** It is 1.4–2.8x
  faster above the crossover and the gap grows with N.
- Your keys are 64-bit and you want to beat parallel radix. PTMS is
  competitive (1.07–1.21x) but IPS4o does better.
- Your inputs are `std::string` or other non-trivially-copyable types.
  Use `tbb::parallel_sort`.
- Your inputs are almost-sorted or have long runs. A natural merge sort
  will do better.

**The honest positioning:** PTMS is a fast sorter for cache-resident
data with a small, portable, OpenMP-only implementation. IPS4o is a
faster sorter for DRAM-bound data with a larger implementation and a
TBB dependency. Both are useful; they occupy different regions of the
size/type space.

---

## License

MIT. See [LICENSE](LICENSE).
