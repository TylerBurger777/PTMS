// PTMS: internal API
// Used by benchmark to run the sort body inside an existing OpenMP
// parallel region (avoids nested parallel regions).
//
// Do NOT include from user code.

#pragma once

namespace ptms {
namespace detail {

// Body of the sort without #pragma omp parallel.
// Must be called from inside an active parallel region, from a single
// thread (e.g. from within #pragma omp single).
void sort_body(int* a, int* buf, int n, int threads);

} // namespace detail
} // namespace ptms
