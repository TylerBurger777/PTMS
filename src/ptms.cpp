// PTMS: Parallel Tournament Merge Sort
// Implementation — optimized version:
//   - ping-pong buffers in recursiveSortPP
//   - threshold check in middleAlgorithm
//   - sorting networks only for n = 4..7
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#include "ptms.h"
#include "ptms_detail.h"

#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace ptms {
namespace {

// =====================================================================
//                     BASE CASE CONFIGURATION
// =====================================================================

constexpr int  BASE_CASE_THRESHOLD = 7;
constexpr bool USE_SIMD_FOR_N8     = true;
constexpr bool USE_SIMD_FOR_N16    = true;

// =====================================================================
//                     ALIGNED MEMORY
// =====================================================================

int* aligned_alloc_int(size_t n) {
    void* p = nullptr;
#ifdef _WIN32
    p = _aligned_malloc(n * sizeof(int), 64);
#else
    if (posix_memalign(&p, 64, n * sizeof(int)) != 0) p = nullptr;
#endif
    return static_cast<int*>(p);
}

void aligned_free_int(int* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// =====================================================================
//                     BRANCHLESS COMPARE-AND-SWAP
// =====================================================================

inline void cmp_swap(int& a, int& b) {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    a = lo;
    b = hi;
}

// =====================================================================
//                     SORTING NETWORKS (n = 4..7)
// =====================================================================

inline void sort4(int* a) {
    cmp_swap(a[0], a[1]); cmp_swap(a[2], a[3]);
    cmp_swap(a[0], a[2]); cmp_swap(a[1], a[3]);
    cmp_swap(a[1], a[2]);
}

inline void sort5(int* a) {
    cmp_swap(a[0], a[1]); cmp_swap(a[3], a[4]);
    cmp_swap(a[2], a[4]);
    cmp_swap(a[2], a[3]); cmp_swap(a[0], a[3]);
    cmp_swap(a[0], a[2]); cmp_swap(a[1], a[4]);
    cmp_swap(a[1], a[3]); cmp_swap(a[1], a[2]);
}

inline void sort6(int* a) {
    cmp_swap(a[1], a[2]); cmp_swap(a[4], a[5]);
    cmp_swap(a[0], a[2]); cmp_swap(a[3], a[5]);
    cmp_swap(a[0], a[1]); cmp_swap(a[3], a[4]);
    cmp_swap(a[2], a[5]);
    cmp_swap(a[0], a[3]); cmp_swap(a[1], a[4]);
    cmp_swap(a[2], a[4]); cmp_swap(a[1], a[3]);
    cmp_swap(a[2], a[3]);
}

inline void sort7(int* a) {
    cmp_swap(a[0], a[1]); cmp_swap(a[2], a[3]); cmp_swap(a[4], a[5]);
    cmp_swap(a[0], a[2]); cmp_swap(a[1], a[3]); cmp_swap(a[4], a[6]);
    cmp_swap(a[1], a[2]); cmp_swap(a[5], a[6]); cmp_swap(a[0], a[4]);
    cmp_swap(a[1], a[5]); cmp_swap(a[2], a[6]); cmp_swap(a[1], a[4]);
    cmp_swap(a[3], a[6]); cmp_swap(a[2], a[4]); cmp_swap(a[3], a[5]);
    cmp_swap(a[3], a[4]);
}

inline void smallSort(int* a, int n) {
    switch (n) {
        case 4: sort4(a); break;
        case 5: sort5(a); break;
        case 6: sort6(a); break;
        case 7: sort7(a); break;
        default: break;
    }
}

// =====================================================================
//                     BRANCHLESS MERGE
// =====================================================================

void mergeBranchless(const int* __restrict__ src,
                     int* __restrict__ dst, int dstLo,
                     int aLo, int aHi, int bLo, int bHi) {
    int i = aLo, j = bLo, k = dstLo;
    while (i <= aHi && j <= bHi) {
        int va = src[i], vb = src[j];
        int takeA = (va <= vb);
        dst[k++] = takeA ? va : vb;
        i += takeA;
        j += !takeA;
    }
    while (i <= aHi) dst[k++] = src[i++];
    while (j <= bHi) dst[k++] = src[j++];
}

// =====================================================================
//                     AVX2 KERNELS
// =====================================================================
#if defined(__AVX2__) || defined(__AVX512F__)

inline __m256i bitonic_merge_internal_8(__m256i v) {
    const __m256i p4 = _mm256_setr_epi32(4,5,6,7,0,1,2,3);
    const __m256i p2 = _mm256_setr_epi32(2,3,0,1,6,7,4,5);
    const __m256i p1 = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    __m256i vp = _mm256_permutevar8x32_epi32(v, p4);
    __m256i lo = _mm256_min_epi32(v, vp), hi = _mm256_max_epi32(v, vp);
    v = _mm256_permute2x128_si256(lo, hi, 0x20);
    vp = _mm256_permutevar8x32_epi32(v, p2);
    lo = _mm256_min_epi32(v, vp); hi = _mm256_max_epi32(v, vp);
    v = _mm256_blend_epi32(lo, hi, 0xCC);
    vp = _mm256_permutevar8x32_epi32(v, p1);
    lo = _mm256_min_epi32(v, vp); hi = _mm256_max_epi32(v, vp);
    v = _mm256_blend_epi32(lo, hi, 0xAA);
    return v;
}

inline __m256i bitonic_sort_8(__m256i v) {
    const __m256i p1 = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    const __m256i p2 = _mm256_setr_epi32(2,3,0,1,6,7,4,5);
    __m256i vp = _mm256_permutevar8x32_epi32(v, p1);
    __m256i lo = _mm256_min_epi32(v, vp), hi = _mm256_max_epi32(v, vp);
    v = _mm256_blend_epi32(lo, hi, 0x66);
    vp = _mm256_permutevar8x32_epi32(v, p2);
    lo = _mm256_min_epi32(v, vp); hi = _mm256_max_epi32(v, vp);
    v = _mm256_blend_epi32(lo, hi, 0x3C);
    vp = _mm256_permutevar8x32_epi32(v, p1);
    lo = _mm256_min_epi32(v, vp); hi = _mm256_max_epi32(v, vp);
    v = _mm256_blend_epi32(lo, hi, 0x5A);
    return bitonic_merge_internal_8(v);
}

inline void bitonic_merge_8x8(__m256i a, __m256i b,
                              __m256i& lo, __m256i& hi) {
    b = _mm256_permutevar8x32_epi32(b, _mm256_setr_epi32(7,6,5,4,3,2,1,0));
    __m256i mn = _mm256_min_epi32(a, b), mx = _mm256_max_epi32(a, b);
    lo = bitonic_merge_internal_8(mn);
    hi = bitonic_merge_internal_8(mx);
}

#if !defined(__AVX512F__)
inline void sort16SIMD(int* a, int lo) {
    __m256i v0 = _mm256_loadu_si256((const __m256i*)(a + lo));
    __m256i v1 = _mm256_loadu_si256((const __m256i*)(a + lo + 8));
    v0 = bitonic_sort_8(v0);
    v1 = bitonic_sort_8(v1);
    __m256i o0, o1;
    bitonic_merge_8x8(v0, v1, o0, o1);
    _mm256_storeu_si256((__m256i*)(a + lo), o0);
    _mm256_storeu_si256((__m256i*)(a + lo + 8), o1);
}
#endif

#endif // AVX2 || AVX512F

// =====================================================================
//                     AVX-512 KERNELS
// =====================================================================
#if defined(__AVX512F__)

inline __m512i stage512(__m512i v, __m512i perm, __mmask16 mask) {
    __m512i vp = _mm512_permutexvar_epi32(perm, v);
    __m512i lo = _mm512_min_epi32(v, vp);
    __m512i hi = _mm512_max_epi32(v, vp);
    return _mm512_mask_blend_epi32(mask, lo, hi);
}

inline __m512i bitonic_merge_internal_16(__m512i v) {
    const __m512i p8 = _mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);
    const __m512i p4 = _mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11);
    const __m512i p2 = _mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
    const __m512i p1 = _mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    v = stage512(v, p8, (__mmask16)0xFF00);
    v = stage512(v, p4, (__mmask16)0xF0F0);
    v = stage512(v, p2, (__mmask16)0xCCCC);
    v = stage512(v, p1, (__mmask16)0xAAAA);
    return v;
}

inline __m512i bitonic_sort_16(__m512i v) {
    const __m512i p1 = _mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    const __m512i p2 = _mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
    const __m512i p4 = _mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11);
    v = stage512(v, p1, (__mmask16)0x6666);
    v = stage512(v, p2, (__mmask16)0x3C3C);
    v = stage512(v, p1, (__mmask16)0x5A5A);
    v = stage512(v, p4, (__mmask16)0x0FF0);
    v = stage512(v, p2, (__mmask16)0x33CC);
    v = stage512(v, p1, (__mmask16)0x55AA);
    return bitonic_merge_internal_16(v);
}

inline void bitonic_merge_16x16(__m512i a, __m512i b,
                                __m512i& lo, __m512i& hi) {
    const __m512i rev = _mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    b = _mm512_permutexvar_epi32(rev, b);
    __m512i mn = _mm512_min_epi32(a, b);
    __m512i mx = _mm512_max_epi32(a, b);
    lo = bitonic_merge_internal_16(mn);
    hi = bitonic_merge_internal_16(mx);
}

#endif // __AVX512F__

// =====================================================================
//                     MERGE DISPATCHER
// =====================================================================

void mergeBase(const int* __restrict__ src, int* __restrict__ dst,
               int dstLo, int aLo, int aHi, int bLo, int bHi) {
    const int na = aHi - aLo + 1;
    const int nb = bHi - bLo + 1;
#if defined(__AVX512F__)
    if (na == 16 && nb == 16) {
        __m512i a = _mm512_loadu_si512((const void*)(src + aLo));
        __m512i b = _mm512_loadu_si512((const void*)(src + bLo));
        __m512i lo, hi;
        bitonic_merge_16x16(a, b, lo, hi);
        _mm512_storeu_si512((void*)(dst + dstLo), lo);
        _mm512_storeu_si512((void*)(dst + dstLo + 16), hi);
        return;
    }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
    if (na == 8 && nb == 8) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(src + aLo));
        __m256i b = _mm256_loadu_si256((const __m256i*)(src + bLo));
        __m256i lo, hi;
        bitonic_merge_8x8(a, b, lo, hi);
        _mm256_storeu_si256((__m256i*)(dst + dstLo), lo);
        _mm256_storeu_si256((__m256i*)(dst + dstLo + 8), hi);
        return;
    }
#endif
    mergeBranchless(src, dst, dstLo, aLo, aHi, bLo, bHi);
}

// =====================================================================
//                     PARALLEL MERGE (Level 1)
// =====================================================================

constexpr int MERGE_SEQ_THRESHOLD = 2048;
constexpr int MERGE_MAX_DEPTH     = 5;

void mergeParallel(const int* __restrict__ src,
                   int* __restrict__ dst, int dstLo,
                   int aLo, int aHi, int bLo, int bHi, int depth) {
    if (aLo > aHi) {
        std::memcpy(dst + dstLo, src + bLo, (bHi - bLo + 1) * sizeof(int));
        return;
    }
    if (bLo > bHi) {
        std::memcpy(dst + dstLo, src + aLo, (aHi - aLo + 1) * sizeof(int));
        return;
    }
    int na = aHi - aLo + 1, nb = bHi - bLo + 1, n = na + nb;
    if (n <= MERGE_SEQ_THRESHOLD || depth >= MERGE_MAX_DEPTH) {
        mergeBase(src, dst, dstLo, aLo, aHi, bLo, bHi);
        return;
    }
    if (na >= nb) {
        int aMid = aLo + na / 2;
        int x = src[aMid];
        int bSplit = (int)(std::lower_bound(src + bLo, src + bHi + 1, x) - src);
        int leftSize = (aMid - aLo) + (bSplit - bLo) + 1;
        #pragma omp task
        mergeParallel(src, dst, dstLo, aLo, aMid, bLo, bSplit - 1, depth + 1);
        mergeParallel(src, dst, dstLo + leftSize, aMid + 1, aHi, bSplit, bHi, depth + 1);
        #pragma omp taskwait
    } else {
        int bMid = bLo + nb / 2;
        int x = src[bMid];
        int aSplit = (int)(std::lower_bound(src + aLo, src + aHi + 1, x) - src);
        int leftSize = (aSplit - aLo) + (bMid - bLo) + 1;
        #pragma omp task
        mergeParallel(src, dst, dstLo, aLo, aSplit - 1, bLo, bMid, depth + 1);
        mergeParallel(src, dst, dstLo + leftSize, aSplit, aHi, bMid + 1, bHi, depth + 1);
        #pragma omp taskwait
    }
}

// =====================================================================
//                     RECURSIVE SORT WITH PING-PONG
// =====================================================================

void recursiveSortPP(int* a, int* buf, int lo, int hi, int* out) {
    int n = hi - lo + 1;
    if (n <= 1) {
        if (out != a) out[lo] = a[lo];
        return;
    }

#if defined(__AVX512F__)
    if (USE_SIMD_FOR_N16 && n == 16) {
        __m512i v = _mm512_loadu_si512((const void*)(a + lo));
        v = bitonic_sort_16(v);
        _mm512_storeu_si512((void*)(a + lo), v);
        if (out != a) std::memcpy(out + lo, a + lo, 16 * sizeof(int));
        return;
    }
#elif defined(__AVX2__)
    if (USE_SIMD_FOR_N16 && n == 16) {
        sort16SIMD(a, lo);
        if (out != a) std::memcpy(out + lo, a + lo, 16 * sizeof(int));
        return;
    }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
    if (USE_SIMD_FOR_N8 && n == 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(a + lo));
        v = bitonic_sort_8(v);
        _mm256_storeu_si256((__m256i*)(a + lo), v);
        if (out != a) std::memcpy(out + lo, a + lo, 8 * sizeof(int));
        return;
    }
#endif

    if (n <= BASE_CASE_THRESHOLD) {
        smallSort(a + lo, n);
        if (out != a) std::memcpy(out + lo, a + lo, n * sizeof(int));
        return;
    }

    int mid = lo + (hi - lo) / 2;
    int* work = (out == a) ? buf : a;

    recursiveSortPP(a, buf, lo, mid, work);
    recursiveSortPP(a, buf, mid + 1, hi, work);

    mergeBase(work, out, lo, lo, mid, mid + 1, hi);
}

void middleAlgorithm(int* a, int* buf, int lo, int hi) {
    if (lo >= hi) return;
    int n = hi - lo + 1;
    if (n <= BASE_CASE_THRESHOLD) {
        smallSort(a + lo, n);
        return;
    }
    int mid = lo + (hi - lo) / 2;
    recursiveSortPP(a, buf, lo, mid, a);
    recursiveSortPP(a, buf, mid + 1, hi, a);
}

// =====================================================================
//                     TOURNAMENT MERGE
// =====================================================================

void tournamentMerge(int* a, int* buf, int N,
                     std::vector<std::pair<int,int>>& parts) {
    int* src = a;
    int* dst = buf;
    while (parts.size() > 1) {
        int n = (int)parts.size();
        int m = n / 2;
        #pragma omp taskgroup
        {
            for (int i = 0; i < m; ++i) {
                int loA = parts[2*i].first,   hiA = parts[2*i].second;
                int loB = parts[2*i+1].first, hiB = parts[2*i+1].second;
                #pragma omp task
                mergeParallel(src, dst, loA, loA, hiA, loB, hiB, 0);
            }
            if (n & 1) {
                int lo = parts.back().first, hi = parts.back().second;
                #pragma omp task
                std::memcpy(dst + lo, src + lo, (hi - lo + 1) * sizeof(int));
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
    if (src != a) std::memcpy(a, src, N * sizeof(int));
}

} // anonymous namespace

// =====================================================================
//                     DETAIL
// =====================================================================

namespace detail {

void sort_body(int* a, int* buf, int N, int THREADS) {
    #pragma omp taskgroup
    {
        for (int t = 0; t < THREADS; ++t) {
            int lo = (int)((long long)t * N / THREADS);
            int hi = (int)((long long)(t + 1) * N / THREADS) - 1;
            if (t == THREADS - 1) hi = N - 1;
            #pragma omp task
            middleAlgorithm(a, buf, lo, hi);
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

    tournamentMerge(a, buf, N, parts);
}

} // namespace detail

// =====================================================================
//                     PUBLIC API
// =====================================================================

void sort_int(int* a, int* buf, int n, int threads) {
    if (n <= 1) return;
    if (a == nullptr || buf == nullptr || buf == a) return;

    int P = (threads > 0) ? threads : omp_get_max_threads();
    if (P > n) P = n;
    if (P < 1) P = 1;

    #pragma omp parallel num_threads(P)
    {
        #pragma omp single
        detail::sort_body(a, buf, n, P);
    }
}

const char* simd_level() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#else
    return "scalar";
#endif
}

} // namespace ptms
