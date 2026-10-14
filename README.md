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
`tbb::parallel_sort` and with parallel radix sort on 64-bit keys.

---

## Contents

- [What it is](#what-it-is)
- [Architecture](#architecture)
- [API](#api)
- [Building](#building)
- [Benchmarks](#benchmarks)
- [Architectural limitations](#architectural-limitations)
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

---

## Architectural limitations

**PTMS is not a state-of-the-art parallel sorter.** It is a
well-optimised 2-way merge sort. Where the field has moved on, PTMS has
not kept up. Concretely:

### 1. It loses to parallel samplesort on every measured type and size

Comparison against [IPS4o](https://github.com/ips4o/ips4o)
(`ips4o::parallel::sort`, BSD-2-Clause), same machine, same run
conditions (Runs = 5, warmup = 2, cooldown = 1000 ms, median):

| Type | N | PTMS | IPS4o | PTMS / IPS4o |
|---|---|---|---|---|
| int32 | 4M | 17.5 ms | 10.2 ms | 0.58x |
| int32 | 16M | 94.3 ms | 66.4 ms | 0.70x |
| int64 | 4M | 32.7 ms | 15.5 ms | 0.47x |
| int64 | 16M | 159.5 ms | 56.9 ms | 0.36x |
| double | 4M | 35.9 ms | 18.0 ms | 0.50x |
| double | 16M | 164.7 ms | 84.2 ms | 0.51x |

PTMS is 1.4–2.8x slower than IPS4o across the board. The gap is largest
on 64-bit keys at 16M, where the memory-bound behaviour of multi-level
merge sort punishes PTMS hardest.

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

Conclusion: naive multi-way loses to 2-way merge, and a proper
sophisticated samplesort (IPS4o) wins at every size measured. There is
no configuration of merge sort that beats IPS4o on uniform random data
at these sizes; the algorithmic constant factors favour samplesort.

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

PTMS is worth using when:

- You need a fast, portable merge sort and cannot bring in a samplesort
  dependency.
- Your keys are 64-bit and you want something competitive with parallel
  radix without paying the `sizeof(T)²` cost.
- Your types are trivially copyable and you want a template that just
  works, without writing per-type kernels.
- You are already using OpenMP and do not want a second threading
  runtime.

PTMS is **not** the right choice when:

- Peak throughput on uniform random data is the only metric that
  matters. Use IPS4o or parallel radix.
- Your inputs are `std::string` or other non-trivially-copyable types.
  Use `tbb::parallel_sort`.
- Your inputs are almost-sorted or have long runs. A natural merge sort
  will do better.

---

## License

MIT. See [LICENSE](LICENSE).
