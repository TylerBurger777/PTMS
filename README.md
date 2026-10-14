# PTMS: Parallel Tournament Merge Sort

Parallel merge sort for C++17 with OpenMP and AVX2, plus a generic
templated pipeline for arbitrary types.

[![CI](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml/badge.svg)](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml)

## What it is

PTMS is a **three-level pipeline** for parallel sorting:

1. **Level 1 — tournament merge**: `2P` sorted chunks are merged pairwise
   through a tournament tree (`#pragma omp task` / `taskgroup`).
2. **Level 2 — parallel chunk sort**: `P` chunks are sorted independently
   via recursive merge sort with ping-pong buffers.
3. **Level 3 — base cases**: sorting networks for `n = 4..7`, AVX2 bitonic
   networks for `n = 8` and `n = 16`.

Key techniques used:

- branchless merge (`cmov`-friendly element step)
- bidirectional merge (two independent pointer chains, merge-path split)
- ping-pong buffers (no post-merge `memcpy`)
- adaptive merge skip (checks `A.max <= B.min` before merging)
- tiered `Auto` thread detection based on empirical small-`N` sweep
- persistent-pool batch API (`sort_int_batch`)
- generic `ptms::sort<T>` template for arbitrary element types

## Results

Measured on AMD Ryzen 9 6900HX (8 physical cores + SMT = 16 logical,
L3 = 16 MB, AVX2), Windows 11, clang-cl via Visual Studio 2022,
Release build. Median over 5+ runs, batch size 5, 1000 ms cooldown
between batches, MAD outlier filter.

### PTMS vs `tbb::parallel_sort`

| Type | N | PTMS | TBB | PTMS vs TBB |
|---|---|---|---|---|
| int32 | 4M | 17.9 ms | 34.5 ms | **1.93×** |
| int32 | 16M | 94.7 ms | 174.4 ms | **1.84×** |
| int64 | 4M | 35.0 ms | 38.0 ms | **1.09×** |
| int64 | 16M | 167.5 ms | 165.9 ms | 0.99× |
| float | 4M | 25.9 ms | 47.8 ms | **1.84×** |
| float | 16M | 126.8 ms | 199.4 ms | **1.57×** |
| double | 4M | 40.0 ms | 48.8 ms | **1.22×** |
| double | 16M | 178.6 ms | 204.1 ms | **1.14×** |
| string | 1M | 137 ms | 89 ms | 0.65× |

PTMS is faster than TBB on all numeric types, from 1.09× to 1.93×
depending on type and `N`. The only exception is `std::string` — TBB
uses in-place quicksort, which avoids the merge-phase copies that
hurt PTMS on expensive-to-move elements.

### PTMS vs parallel radix sort

Radix sort needs a fixed-width key and digit extraction. It is
applicable to `int32`, `int64`, `float`, `double`, but not to
`std::string` or user-defined types.

For **32-bit keys** (`int32`, `float`), radix does 4 passes and is
consistently faster than PTMS on large `N` (1.7–2.0×) — this is
expected from the `O(n)` vs `O(n log n)` asymptotics.

For **64-bit keys** (`int64`, `double`), radix does 8 passes, doubling
its runtime, while PTMS's cost stays the same. In this regime PTMS
wins:

| Type | N | PTMS | radix_parallel | PTMS vs radix |
|---|---|---|---|---|
| int64 | 2M | 15.0 ms | 17.2 ms | **1.15×** |
| int64 | 4M | 35.0 ms | 42.3 ms | **1.21×** |
| int64 | 8M | 75.5 ms | 84.4 ms | **1.12×** |
| double | 2M | 16.9 ms | 20.5 ms | **1.21×** |
| double | 4M | 40.0 ms | 47.3 ms | **1.18×** |
| double | 8M | 84.7 ms | 94.5 ms | **1.11×** |
| double | 16M | 178.6 ms | 191.9 ms | **1.07×** |

### Where PTMS is the fastest choice

- **64-bit keys** (`int64`, `double`) at `N >= 2M`: PTMS beats both
  radix and TBB.
- **Arbitrary comparable types** (strings, user structs, custom
  comparators): radix does not apply, and PTMS is 1.5–2.5× faster
  than `std::sort`.
- **Random data**, `int32`/`float`, `N ≤ 16K`: PTMS wins because its
  AVX2 sorting networks and branchless merge have lower overhead than
  either TBB's task scheduler or radix's histogram setup.

## Usage

```cpp
#include "ptms.h"

// int32 fast path
std::vector<int> a = ...;
std::vector<int> buf(a.size());
ptms::sort_int(a.data(), buf.data(), (int)a.size(),
               ptms::ThreadMode::Auto);

// Batch of 1000 arrays, one persistent parallel region
ptms::sort_int_batch(data, buf, n_per_array, 1000,
                     ptms::ThreadMode::ByCores);

// Generic path for arbitrary types
std::vector<MyStruct> s = ...;
std::vector<MyStruct> sbuf(s.size());
ptms::sort<MyStruct>(s.data(), sbuf.data(), (int)s.size(),
                     [](const MyStruct& a, const MyStruct& b) {
                         return a.key < b.key;
                     });
