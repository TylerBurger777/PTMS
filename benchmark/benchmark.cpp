// PTMS: benchmark with self-tests
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#include "ptms.h"
#include "ptms_detail.h"

#include <omp.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cmath>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// =====================================================================
//                     SELF-TESTS (independent copies of kernels)
// =====================================================================

static inline void cmp_swap_t(int& a, int& b) {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    a = lo;
    b = hi;
}

static inline void t_sort4(int* a) {
    cmp_swap_t(a[0], a[1]); cmp_swap_t(a[2], a[3]);
    cmp_swap_t(a[0], a[2]); cmp_swap_t(a[1], a[3]);
    cmp_swap_t(a[1], a[2]);
}

static inline void t_sort5(int* a) {
    cmp_swap_t(a[0], a[1]); cmp_swap_t(a[3], a[4]);
    cmp_swap_t(a[2], a[4]);
    cmp_swap_t(a[2], a[3]); cmp_swap_t(a[0], a[3]);
    cmp_swap_t(a[0], a[2]); cmp_swap_t(a[1], a[4]);
    cmp_swap_t(a[1], a[3]); cmp_swap_t(a[1], a[2]);
}

static inline void t_sort6(int* a) {
    cmp_swap_t(a[1], a[2]); cmp_swap_t(a[4], a[5]);
    cmp_swap_t(a[0], a[2]); cmp_swap_t(a[3], a[5]);
    cmp_swap_t(a[0], a[1]); cmp_swap_t(a[3], a[4]);
    cmp_swap_t(a[2], a[5]);
    cmp_swap_t(a[0], a[3]); cmp_swap_t(a[1], a[4]);
    cmp_swap_t(a[2], a[4]); cmp_swap_t(a[1], a[3]);
    cmp_swap_t(a[2], a[3]);
}

static inline void t_sort7(int* a) {
    cmp_swap_t(a[0], a[1]); cmp_swap_t(a[2], a[3]); cmp_swap_t(a[4], a[5]);
    cmp_swap_t(a[0], a[2]); cmp_swap_t(a[1], a[3]); cmp_swap_t(a[4], a[6]);
    cmp_swap_t(a[1], a[2]); cmp_swap_t(a[5], a[6]); cmp_swap_t(a[0], a[4]);
    cmp_swap_t(a[1], a[5]); cmp_swap_t(a[2], a[6]); cmp_swap_t(a[1], a[4]);
    cmp_swap_t(a[3], a[6]); cmp_swap_t(a[2], a[4]); cmp_swap_t(a[3], a[5]);
    cmp_swap_t(a[3], a[4]);
}

static bool selfTestNetworks(int iters = 5000) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(-10000, 10000);
    bool ok = true;
    for (int n = 4; n <= 7; ++n) {
        for (int iter = 0; iter < iters; ++iter) {
            alignas(64) int v[16];
            std::vector<int> ref(n);
            for (int i = 0; i < n; ++i) { v[i] = d(rng); ref[i] = v[i]; }

            switch (n) {
                case 4: t_sort4(v); break;
                case 5: t_sort5(v); break;
                case 6: t_sort6(v); break;
                case 7: t_sort7(v); break;
            }
            std::sort(ref.begin(), ref.end());
            for (int i = 0; i < n; ++i)
                if (v[i] != ref[i]) {
                    std::cerr << "sort" << n << " FAIL\n"; return false;
                }
        }
        std::cout << "Self-test sort" << (n < 10 ? " " : "") << n
                  << " network : PASS\n";
    }
    return ok;
}

static bool selfTestAll() {
    bool ok = true;
    if (!selfTestNetworks()) ok = false;
    std::cout << "SIMD level     : " << ptms::simd_level() << "\n";
    return ok;
}

// =====================================================================
//                     STATISTICS
// =====================================================================

struct Stats { double best, worst, median, trimmed, mean, stddev; };

static Stats computeStats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.best = v.front(); s.worst = v.back();
    s.median = v[v.size() / 2];
    int trim = (int)v.size() * 5 / 100;
    double sum = 0;
    for (int i = trim; i < (int)v.size() - trim; ++i) sum += v[i];
    s.trimmed = sum / (v.size() - 2 * trim);
    double total = 0;
    for (double t : v) total += t;
    s.mean = total / v.size();
    double var = 0;
    for (double t : v) var += (t - s.mean) * (t - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}

// =====================================================================
//                     MAIN
// =====================================================================
int main() {
    std::cout << "=== PTMS: Parallel Tournament Merge Sort ===\n";
    std::cout << "Build          : "
#ifdef NDEBUG
              << "RELEASE"
#else
              << "DEBUG"
#endif
              << "\n";

    if (!selfTestAll()) return 1;
    std::cout << "\n";

    const int N = 4'000'000;
    const int WARMUP_RUNS   = 400;
    const int MEASURED_RUNS = 600;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);

    std::vector<int> a(N);
    for (auto& x : a) x = dist(rng);
    std::vector<int> reference = a;
    std::sort(reference.begin(), reference.end());

    std::vector<int> work(N), scratch(N);

    int THREADS = omp_get_max_threads();
    if (THREADS > N) THREADS = N;

    if (N < 4 * THREADS) {
        std::cerr << "WARNING: N < 4 * THREADS; correctness not guaranteed.\n";
    }

    std::cout << "N = " << N << ", threads = " << THREADS << "\n";
    std::cout << "Warmup: " << WARMUP_RUNS
              << ", Measured: " << MEASURED_RUNS << "\n\n";

    std::vector<double> times;
    times.reserve(MEASURED_RUNS);

    #pragma omp parallel num_threads(THREADS)
    {
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            #pragma omp single
            {
                std::memcpy(work.data(), a.data(), N * sizeof(int));
                ptms::detail::sort_body(work.data(), scratch.data(), N, THREADS);
            }
        }

        for (int run = 0; run < MEASURED_RUNS; ++run) {
            #pragma omp single
            {
                std::memcpy(work.data(), a.data(), N * sizeof(int));
                auto t0 = std::chrono::high_resolution_clock::now();
                ptms::detail::sort_body(work.data(), scratch.data(), N, THREADS);
                auto t1 = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double>(t1 - t0).count());

                if (run % 200 == 0 || run == MEASURED_RUNS - 1) {
                    bool ok = std::equal(work.begin(), work.end(), reference.begin());
                    std::cout << "  [check at run " << WARMUP_RUNS + run + 1
                              << "] correct = " << std::boolalpha << ok << "\n";
                }
            }
        }
    }

    Stats ptms = computeStats(times);

    std::cout << "\n=================== PTMS RESULTS ===================\n";
    std::cout << "Best            : " << ptms.best    << " s\n";
    std::cout << "Worst           : " << ptms.worst   << " s\n";
    std::cout << "Median          : " << ptms.median  << " s\n";
    std::cout << "Trimmed mean 5% : " << ptms.trimmed << " s\n";
    std::cout << "Mean (raw)      : " << ptms.mean    << " s\n";
    std::cout << "Stddev          : " << ptms.stddev  << " s\n";
    std::cout << "====================================================\n";

    return 0;
}
