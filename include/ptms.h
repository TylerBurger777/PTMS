// PTMS: Parallel Tournament Merge Sort
// Public API header
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace ptms {

// Sort an array of int in ascending order using the PTMS algorithm.
//
// Parameters:
//   a       - input/output array of int (size >= n)
//   buf     - auxiliary buffer (size >= n), must not alias 'a'
//   n       - number of elements
//   threads - number of OpenMP threads; if 0, uses all available
//
// Requirements:
//   - buf != a, and both are non-null
//   - buf has at least n elements
//   - n >= 4 * threads (for correctness with small blocks)
//
// Not thread-safe: do not call concurrently on overlapping data.
void sort_int(int* a, int* buf, int n, int threads = 0);

// Return the SIMD level compiled into this binary.
// Possible values: "AVX-512", "AVX2", "scalar".
const char* simd_level();

} // namespace ptms
