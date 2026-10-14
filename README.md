# PTMS: Parallel Tournament Merge Sort

[![CI](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml/badge.svg)](https://github.com/TylerBurger777/PTMS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance parallel merge sort with SIMD acceleration
(AVX2 / AVX-512) and OpenMP tasks.

## Overview

PTMS is a parallel sorting algorithm built as a three-tier pipeline:

1. **Tier 1 — Tournament merge:** 2P sorted parts are merged pairwise
   level by level (2P → P → P/2 → ... → 1). Each level runs as
   parallel OpenMP tasks.
2. **Tier 2 — Parallel sort of P chunks:** the array is split into P
   equal chunks, each sorted independently via recursive merge sort
   with ping-pong buffers.
3. **Tier 3 — Base cases:** sorting networks for blocks of 4–7
   elements; SIMD bitonic sort for blocks of 8 and 16 (AVX2 / AVX-512).

Merges inside tasks use **parallel merge with binary splitting**,
which removes the Amdahl bottleneck on the final merge.

## Performance

Benchmark on 4 million `int` (16 MB, cache-resident) on a 16-thread
AMD Ryzen 9 6900HX:

| Algorithm              | Median (s) | Speedup vs `std::sort` |
|------------------------|-----------:|----------------------:|
| `std::sort` (1 thread) |      0.300 |                 1.00x |
| `tbb::parallel_sort`   |      0.036 |                 8.30x |
| **PTMS**               |  **0.0163** |            **~18.4x** |

## Thread modes

PTMS supports four ways to pick the number of threads:

| Mode | Description |
|---|---|
| `Auto` | Automatically by array size and core count. Default. |
| `Single` | Always 1 thread (sequential). |
| `ByCores` | Physical core count (no SMT). |
| `Fixed(N)` | Exactly N threads. |

When `N < 10000` (see `PARALLEL_THRESHOLD` in `src/ptms.cpp`),
parallelism is disabled automatically, since the OpenMP task overhead
exceeds the parallel speedup. `Fixed` bypasses this check.

```cpp
#include <ptms.h>

std::vector<int> a = ...;
std::vector<int> buf(a.size());

ptms::sort_int(a.data(), buf.data(), (int)a.size());                         // Auto
ptms::sort_int(a.data(), buf.data(), (int)a.size(), ptms::ThreadMode::Single);
ptms::sort_int(a.data(), buf.data(), (int)a.size(), ptms::ThreadMode::ByCores);
ptms::sort_int(a.data(), buf.data(), (int)a.size(), ptms::ThreadMode::Fixed, 8);
