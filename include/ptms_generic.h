// PTMS: Parallel Tournament Merge Sort
// Generic templated pipeline for arbitrary element types.
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#pragma once

#include "ptms.h"

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ptms {
namespace detail {

// =====================================================================
//                     THREAD HELPERS
// =====================================================================

inline int available_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int resolve_thread_count_generic(int n, ThreadMode mode, int fixed_threads) {
    int max_t = available_threads();
    switch (mode) {
        case ThreadMode::Single: return 1;
        case ThreadMode::ByCores: {
            if (n < 512) return 1;
            int phys = physical_core_count();
            return std::max(1, std::min(phys, max_t));
        }
        case ThreadMode::Fixed:
            return std::max(1, std::min(fixed_threads > 0 ? fixed_threads : max_t, max_t));
        case ThreadMode::Auto:
        default: {
            int p;
            if      (n <    512) p = 1;
            else if (n <   2048) p = 2;
            else if (n <   8192) p = 4;
            else if (n <  32768) p = 8;
            else                 p = max_t;
            return std::max(1, std::min(p, max_t));
        }
    }
}

constexpr int BASE_CASE_THRESHOLD_GEN = 7;
constexpr int BIDIRECTIONAL_MIN_TOTAL_GEN = 256;

// =====================================================================
//                     COMPARE-AND-SWAP
// =====================================================================

template <typename T, typename Cmp>
inline void cmp_swap(T& a, T& b, Cmp cmp) {
    if (cmp(b, a)) std::swap(a, b);
}

// =====================================================================
//                     SORTING NETWORKS
// =====================================================================

template <typename T, typename Cmp>
inline void sort4_gen(T* a, Cmp cmp) {
    cmp_swap(a[0], a[1], cmp); cmp_swap(a[2], a[3], cmp);
    cmp_swap(a[0], a[2], cmp); cmp_swap(a[1], a[3], cmp);
    cmp_swap(a[1], a[2], cmp);
}
template <typename T, typename Cmp>
inline void sort5_gen(T* a, Cmp cmp) {
    cmp_swap(a[0], a[1], cmp); cmp_swap(a[3], a[4], cmp);
    cmp_swap(a[2], a[4], cmp); cmp_swap(a[2], a[3], cmp);
    cmp_swap(a[0], a[3], cmp); cmp_swap(a[0], a[2], cmp);
    cmp_swap(a[1], a[4], cmp); cmp_swap(a[1], a[3], cmp);
    cmp_swap(a[1], a[2], cmp);
}
template <typename T, typename Cmp>
inline void sort6_gen(T* a, Cmp cmp) {
    cmp_swap(a[1], a[2], cmp); cmp_swap(a[4], a[5], cmp);
    cmp_swap(a[0], a[2], cmp); cmp_swap(a[3], a[5], cmp);
    cmp_swap(a[0], a[1], cmp); cmp_swap(a[3], a[4], cmp);
    cmp_swap(a[2], a[5], cmp); cmp_swap(a[0], a[3], cmp);
    cmp_swap(a[1], a[4], cmp); cmp_swap(a[2], a[4], cmp);
    cmp_swap(a[1], a[3], cmp); cmp_swap(a[2], a[3], cmp);
}
template <typename T, typename Cmp>
inline void sort7_gen(T* a, Cmp cmp) {
    cmp_swap(a[0], a[1], cmp); cmp_swap(a[2], a[3], cmp); cmp_swap(a[4], a[5], cmp);
    cmp_swap(a[0], a[2], cmp); cmp_swap(a[1], a[3], cmp); cmp_swap(a[4], a[6], cmp);
    cmp_swap(a[1], a[2], cmp); cmp_swap(a[5], a[6], cmp); cmp_swap(a[0], a[4], cmp);
    cmp_swap(a[1], a[5], cmp); cmp_swap(a[2], a[6], cmp); cmp_swap(a[1], a[4], cmp);
    cmp_swap(a[3], a[6], cmp); cmp_swap(a[2], a[4], cmp); cmp_swap(a[3], a[5], cmp);
    cmp_swap(a[3], a[4], cmp);
}
template <typename T, typename Cmp>
inline void smallSort_gen(T* a, int n, Cmp cmp) {
    switch (n) {
        case 4: sort4_gen(a, cmp); break;
        case 5: sort5_gen(a, cmp); break;
        case 6: sort6_gen(a, cmp); break;
        case 7: sort7_gen(a, cmp); break;
        default: break;
    }
}

// =====================================================================
//                     LINEAR MERGE
// =====================================================================

template <typename T, typename Cmp>
void mergeBranchless_gen(const T* __restrict__ src,
                         T* __restrict__ dst, int dstLo,
                         int aLo, int aHi, int bLo, int bHi, Cmp cmp) {
    int i = aLo, j = bLo, k = dstLo;
    while (i <= aHi && j <= bHi) {
        if (cmp(src[j], src[i])) dst[k++] = src[j++];
        else                     dst[k++] = src[i++];
    }
    while (i <= aHi) dst[k++] = src[i++];
    while (j <= bHi) dst[k++] = src[j++];
}

// =====================================================================
//                     BIDIRECTIONAL MERGE (generic, branchless)
// =====================================================================
//
// Two independent pointer chains (forward for left half, backward for
// right half). Element step is cmov-friendly. No block skip.

template <typename T, typename Cmp>
void mergeBidirectional_gen(const T* __restrict__ src,
                            T* __restrict__ dst, int dstLo,
                            int aLo, int aHi, int bLo, int bHi, Cmp cmp) {
    const int na    = aHi - aLo + 1;
    const int nb    = bHi - bLo + 1;
    const int total = na + nb;
    const int half  = total / 2;

    int lo = std::max(0, half - nb);
    int hi = std::min(na, half);
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        const int j   = half - mid;
        if (j > 0 && cmp(src[bLo + j - 1], src[aLo + mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const int i_meet = lo;
    const int j_meet = half - i_meet;

    // ---- Left half, forward ----
    {
        int i = aLo, j = bLo, k = dstLo;
        const int i_end = aLo + i_meet;
        const int j_end = bLo + j_meet;

        while (i < i_end && j < j_end) {
            // A wins on tie: takeA = !cmp(src[j], src[i]).
            int takeA = !cmp(src[j], src[i]);
            dst[k++] = takeA ? src[i] : src[j];
            i += takeA;
            j += !takeA;
        }
        while (i < i_end) dst[k++] = src[i++];
        while (j < j_end) dst[k++] = src[j++];
    }

    // ---- Right half, backward ----
    {
        int i = aHi, j = bHi, k = dstLo + total - 1;
        const int i_end = aLo + i_meet - 1;
        const int j_end = bLo + j_meet - 1;

        while (i > i_end && j > j_end) {
            // A wins on tie: takeA = cmp(src[j], src[i]).
            int takeA = cmp(src[j], src[i]);
            dst[k--] = takeA ? src[i] : src[j];
            i -= takeA;
            j -= !takeA;
        }
        while (i > i_end) dst[k--] = src[i--];
        while (j > j_end) dst[k--] = src[j--];
    }
}

// =====================================================================
//                     MERGE DISPATCHER
// =====================================================================

template <typename T, typename Cmp>
void mergeBase_gen(const T* src, T* dst, int dstLo,
                   int aLo, int aHi, int bLo, int bHi, Cmp cmp) {
    if (!cmp(src[bLo], src[aHi])) {
        const int na = aHi - aLo + 1;
        std::copy(src + aLo, src + aHi + 1, dst + dstLo);
        std::copy(src + bLo, src + bHi + 1, dst + dstLo + na);
        return;
    }
    if (cmp(src[bHi], src[aLo])) {
        const int na = aHi - aLo + 1;
        const int nb = bHi - bLo + 1;
        std::copy(src + bLo, src + bHi + 1, dst + dstLo);
        std::copy(src + aLo, src + aHi + 1, dst + dstLo + nb);
        return;
    }

    const int total = (aHi - aLo + 1) + (bHi - bLo + 1);

    if (total >= BIDIRECTIONAL_MIN_TOTAL_GEN) {
        mergeBidirectional_gen(src, dst, dstLo, aLo, aHi, bLo, bHi, cmp);
        return;
    }
    mergeBranchless_gen(src, dst, dstLo, aLo, aHi, bLo, bHi, cmp);
}

// =====================================================================
//                     PARALLEL MERGE
// =====================================================================

constexpr int MERGE_SEQ_THRESHOLD_GEN = 2048;
constexpr int MERGE_MAX_DEPTH_GEN     = 5;

template <typename T, typename Cmp>
void mergeParallel_gen(const T* __restrict__ src,
                       T* __restrict__ dst, int dstLo,
                       int aLo, int aHi, int bLo, int bHi,
                       int depth, Cmp cmp) {
    if (aLo > aHi) { std::copy(src + bLo, src + bHi + 1, dst + dstLo); return; }
    if (bLo > bHi) { std::copy(src + aLo, src + aHi + 1, dst + dstLo); return; }

    if (!cmp(src[bLo], src[aHi])) {
        const int na = aHi - aLo + 1;
        std::copy(src + aLo, src + aHi + 1, dst + dstLo);
        std::copy(src + bLo, src + bHi + 1, dst + dstLo + na);
        return;
    }
    if (cmp(src[bHi], src[aLo])) {
        const int na = aHi - aLo + 1;
        const int nb = bHi - bLo + 1;
        std::copy(src + bLo, src + bHi + 1, dst + dstLo);
        std::copy(src + aLo, src + aHi + 1, dst + dstLo + nb);
        return;
    }

    int na = aHi - aLo + 1, nb = bHi - bLo + 1, n = na + nb;
    if (n <= MERGE_SEQ_THRESHOLD_GEN || depth >= MERGE_MAX_DEPTH_GEN) {
        mergeBase_gen(src, dst, dstLo, aLo, aHi, bLo, bHi, cmp);
        return;
    }

#ifdef _OPENMP
    #pragma omp task
#endif
    if (na >= nb) {
        int aMid = aLo + na / 2;
        const T& x = src[aMid];
        int bSplit = (int)(std::lower_bound(src + bLo, src + bHi + 1, x, cmp) - src);
        int leftSize = (aMid - aLo) + (bSplit - bLo) + 1;
        mergeParallel_gen(src, dst, dstLo, aLo, aMid, bLo, bSplit - 1, depth + 1, cmp);
        mergeParallel_gen(src, dst, dstLo + leftSize, aMid + 1, aHi, bSplit, bHi, depth + 1, cmp);
#ifdef _OPENMP
        #pragma omp taskwait
#endif
    } else {
        int bMid = bLo + nb / 2;
        const T& x = src[bMid];
        int aSplit = (int)(std::lower_bound(src + aLo, src + aHi + 1, x, cmp) - src);
        int leftSize = (aSplit - aLo) + (bMid - bLo) + 1;
        mergeParallel_gen(src, dst, dstLo, aLo, aSplit - 1, bLo, bMid, depth + 1, cmp);
        mergeParallel_gen(src, dst, dstLo + leftSize, aSplit, aHi, bMid + 1, bHi, depth + 1, cmp);
#ifdef _OPENMP
        #pragma omp taskwait
#endif
    }
}

// =====================================================================
//                     RECURSIVE SORT
// =====================================================================

template <typename T, typename Cmp>
void recursiveSortPP_gen(T* a, T* buf, int lo, int hi, T* out, Cmp cmp) {
    int n = hi - lo + 1;
    if (n <= 1) { if (out != a) out[lo] = a[lo]; return; }
    if (n <= BASE_CASE_THRESHOLD_GEN) {
        smallSort_gen(a + lo, n, cmp);
        if (out != a) std::copy(a + lo, a + lo + n, out + lo);
        return;
    }
    if (std::is_sorted(a + lo, a + hi + 1, cmp)) {
        if (out != a) std::copy(a + lo, a + hi + 1, out + lo);
        return;
    }
    int mid = lo + (hi - lo) / 2;
    T* work = (out == a) ? buf : a;
    recursiveSortPP_gen(a, buf, lo, mid, work, cmp);
    recursiveSortPP_gen(a, buf, mid + 1, hi, work, cmp);
    mergeBase_gen(work, out, lo, lo, mid, mid + 1, hi, cmp);
}

template <typename T, typename Cmp>
void middleAlgorithm_gen(T* a, T* buf, int lo, int hi, Cmp cmp) {
    if (lo >= hi) return;
    int n = hi - lo + 1;
    if (n <= BASE_CASE_THRESHOLD_GEN) { smallSort_gen(a + lo, n, cmp); return; }
    if (std::is_sorted(a + lo, a + hi + 1, cmp)) return;
    int mid = lo + (hi - lo) / 2;
    recursiveSortPP_gen(a, buf, lo, mid, a, cmp);
    recursiveSortPP_gen(a, buf, mid + 1, hi, a, cmp);
}

// =====================================================================
//                     TOURNAMENT MERGE
// =====================================================================

template <typename T, typename Cmp>
void tournamentMerge_gen(T* a, T* buf, int N,
                         std::vector<std::pair<int,int>>& parts, Cmp cmp) {
    T* src = a;
    T* dst = buf;
    while (parts.size() > 1) {
        int n = (int)parts.size();
        int m = n / 2;
#ifdef _OPENMP
        #pragma omp taskgroup
#endif
        {
            for (int i = 0; i < m; ++i) {
                int loA = parts[2*i].first,   hiA = parts[2*i].second;
                int loB = parts[2*i+1].first, hiB = parts[2*i+1].second;
#ifdef _OPENMP
                #pragma omp task
#endif
                mergeParallel_gen(src, dst, loA, loA, hiA, loB, hiB, 0, cmp);
            }
            if (n & 1) {
                int lo = parts.back().first, hi = parts.back().second;
#ifdef _OPENMP
                #pragma omp task
#endif
                std::copy(src + lo, src + hi + 1, dst + lo);
            }
        }
        std::vector<std::pair<int,int>> next;
        next.reserve(m + (n & 1));
        for (int i = 0; i < m; ++i)
            next.push_back({parts[2*i].first, parts[2*i+1].second});
        if (n & 1) next.push_back(parts.back());
        parts.swap(next);
        std::swap(src, dst);
    }
    if (src != a) std::copy(src, src + N, a);
}

// =====================================================================
//                     SORT BODY
// =====================================================================

template <typename T, typename Cmp>
void sort_body_gen(T* a, T* buf, int N, int THREADS, Cmp cmp) {
#ifdef _OPENMP
    #pragma omp taskgroup
#endif
    {
        for (int t = 0; t < THREADS; ++t) {
            int lo = (int)((long long)t * N / THREADS);
            int hi = (int)((long long)(t + 1) * N / THREADS) - 1;
            if (t == THREADS - 1) hi = N - 1;
#ifdef _OPENMP
            #pragma omp task
#endif
            middleAlgorithm_gen(a, buf, lo, hi, cmp);
        }
    }
    std::vector<std::pair<int,int>> parts;
    parts.reserve(2 * THREADS);
    for (int t = 0; t < THREADS; ++t) {
        int lo = (int)((long long)t * N / THREADS);
        int hi = (int)((long long)(t + 1) * N / THREADS) - 1;
        if (t == THREADS - 1) hi = N - 1;
        int mid = lo + (hi - lo) / 2;
        parts.push_back({lo, mid});
        parts.push_back({mid + 1, hi});
    }
    tournamentMerge_gen(a, buf, N, parts, cmp);
}

template <typename T, typename Cmp>
void insertionFallback_gen(T* a, int n, Cmp cmp) {
    for (int i = 1; i < n; ++i) {
        T key = std::move(a[i]);
        int j = i - 1;
        while (j >= 0 && cmp(key, a[j])) {
            a[j + 1] = std::move(a[j]);
            --j;
        }
        a[j + 1] = std::move(key);
    }
}

} // namespace detail

// =====================================================================
//                     PUBLIC TEMPLATE API
// =====================================================================

template <typename T, typename Cmp>
void sort(T* a, T* buf, int n, Cmp cmp,
          ThreadMode mode = ThreadMode::Auto,
          int fixed_threads = 0) {
    if (n <= 1) return;
    if (a == nullptr || buf == nullptr || a == buf) return;
    if (n < 4) { detail::insertionFallback_gen(a, n, cmp); return; }

    int P = detail::resolve_thread_count_generic(n, mode, fixed_threads);
    if (P == 1) { detail::sort_body_gen(a, buf, n, 1, cmp); return; }

#ifdef _OPENMP
    #pragma omp parallel num_threads(P)
    {
        #pragma omp single
        detail::sort_body_gen(a, buf, n, P, cmp);
    }
#else
    detail::sort_body_gen(a, buf, n, 1, cmp);
#endif
}

template <typename T>
void sort(T* a, T* buf, int n,
          ThreadMode mode = ThreadMode::Auto,
          int fixed_threads = 0) {
    sort<T, std::less<T>>(a, buf, n, std::less<T>{}, mode, fixed_threads);
}

template <>
inline void sort<int>(int* a, int* buf, int n,
                      ThreadMode mode, int fixed_threads) {
    sort_int(a, buf, n, mode, fixed_threads);
}

} // namespace ptms
