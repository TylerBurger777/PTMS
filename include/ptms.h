// PTMS: Parallel Tournament Merge Sort
// Public API
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#pragma once

namespace ptms {

// Controls how sort_int chooses the number of threads.
enum class ThreadMode {
    Auto,      // auto-detect from array size and available cores
    Single,    // always single-threaded
    ByCores,   // use physical cores only (no SMT)
    Fixed      // use exactly `fixed_threads` threads (1..max)
};

// One-shot sort of int32. AVX2-accelerated fast path.
void sort_int(int* a, int* buf, int n,
              ThreadMode mode = ThreadMode::Auto,
              int fixed_threads = 0);

// Batch sort of int32. Reuses a single parallel region for all arrays.
void sort_int_batch(int* data, int* buf, int n, int num_arrays,
                    ThreadMode mode = ThreadMode::Auto,
                    int fixed_threads = 0);

// Returns "AVX-512", "AVX2" or "scalar".
const char* simd_level();

// Number of physical cores (excluding SMT siblings).
int physical_core_count();

} // namespace ptms

// Generic templated pipeline for arbitrary types.
// Included last so that ThreadMode is already declared above.
#include "ptms_generic.h"
