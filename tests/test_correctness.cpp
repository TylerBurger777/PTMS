// PTMS: correctness tests
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT
//
// Test coverage:
//   - tiny sizes (n < 4) hitting the insertion-sort fallback
//   - sorting-network boundaries (4..17)
//   - powers of two +/- 1 (2^4-1 .. 2^14+1)
//   - tiered Auto switch points (512, 2048, 8192, 32768)
//   - every ThreadMode, including all Fixed(P) for P = 1..16
//   - odd n with non-power-of-two P (carry-over branch in tournament)
//   - data patterns: random, sorted, reverse, all-equal, almost-sorted,
//     INT_MIN/INT_MAX extremes, two distinct values
//   - sort_int_batch across n, thread modes, and batch sizes
//   - buffer reuse across many consecutive calls
//   - idempotency on already-sorted input
//
// Arrays of size 0..3 are handled by an internal insertion-sort fallback
// (see README "Limitations").

#include "ptms.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// =====================================================================
//                       HELPERS
// =====================================================================

enum class Pattern {
    Random,
    Sorted,
    Reverse,
    AllEqual,
    AlmostSorted,
    Extremes,
    TwoValues
};

static const char* pattern_name(Pattern p) {
    switch (p) {
        case Pattern::Random:       return "random";
        case Pattern::Sorted:       return "sorted";
        case Pattern::Reverse:      return "reverse";
        case Pattern::AllEqual:     return "all-equal";
        case Pattern::AlmostSorted: return "almost-sorted";
        case Pattern::Extremes:     return "extremes";
        case Pattern::TwoValues:    return "two-values";
    }
    return "?";
}

static std::vector<int> make_data(int n, Pattern p, uint32_t seed) {
    std::vector<int> v(n);
    if (n == 0) return v;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(-10000, 10000);

    switch (p) {
        case Pattern::Random:
            for (auto& x : v) x = d(rng);
            break;

        case Pattern::Sorted:
            for (int i = 0; i < n; ++i) v[i] = i;
            break;

        case Pattern::Reverse:
            for (int i = 0; i < n; ++i) v[i] = n - i;
            break;

        case Pattern::AllEqual:
            for (auto& x : v) x = 42;
            break;

        case Pattern::AlmostSorted:
            for (int i = 0; i < n; ++i) v[i] = i;
            for (int s = 0; s < 5 && n > 1; ++s) {
                int i = (int)(rng() % (uint32_t)n);
                int j = (int)(rng() % (uint32_t)n);
                std::swap(v[i], v[j]);
            }
            break;

        case Pattern::Extremes: {
            std::uniform_int_distribution<int> pick(0, 4);
            for (auto& x : v) {
                switch (pick(rng)) {
                    case 0:  x = INT_MIN; break;
                    case 1:  x = INT_MAX; break;
                    case 2:  x = 0;       break;
                    case 3:  x = -1;      break;
                    default: x = d(rng);  break;
                }
            }
            break;
        }

        case Pattern::TwoValues:
            for (auto& x : v) x = (rng() & 1u) ? 7 : 13;
            break;
    }
    return v;
}

// Verify sort_int on the given data. Returns true on success.
static bool testCase(const std::vector<int>& data, const std::string& name,
                     ptms::ThreadMode mode = ptms::ThreadMode::Auto,
                     int fixed = 0) {
    std::vector<int> a = data;
    std::vector<int> buf(std::max<size_t>(1, a.size()));
    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());

    ptms::sort_int(a.data(), buf.data(), (int)a.size(), mode, fixed);

    bool ok = (a == expected);
    if (!ok) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != expected[i]) {
                std::cout << "    first mismatch at index " << i
                          << ": got " << a[i]
                          << ", expected " << expected[i] << "\n";
                break;
            }
        }
    }
    std::cout << "[ " << (ok ? "OK" : "FAIL") << " ] " << name
              << " (n = " << a.size() << ")\n";
    return ok;
}

// Verify sort_int_batch sorts every array in the batch correctly.
static bool testBatch(const std::string& name,
                      int n, int num_arrays,
                      ptms::ThreadMode mode = ptms::ThreadMode::Auto,
                      int fixed = 0) {
    if (n <= 0 || num_arrays <= 0) return true;

    const size_t total = (size_t)n * num_arrays;
    std::vector<int> data(total);
    std::vector<int> buf(total);

    std::mt19937 rng(0xBEEFu + (uint32_t)n + (uint32_t)num_arrays);
    std::uniform_int_distribution<int> d(-100000, 100000);
    for (auto& x : data) x = d(rng);

    ptms::sort_int_batch(data.data(), buf.data(), n, num_arrays, mode, fixed);

    bool ok = true;
    for (int i = 0; i < num_arrays && ok; ++i) {
        std::vector<int> ref(data.begin() + (size_t)i * n,
                             data.begin() + (size_t)(i + 1) * n);
        std::sort(ref.begin(), ref.end());
        for (int j = 0; j < n; ++j) {
            if (data[(size_t)i * n + j] != ref[j]) {
                ok = false;
                std::cout << "    batch[" << i << "][" << j << "] mismatch: "
                          << data[(size_t)i * n + j]
                          << " vs " << ref[j] << "\n";
                break;
            }
        }
    }
    std::cout << "[ " << (ok ? "OK" : "FAIL") << " ] " << name
              << " (n = " << n << ", arrays = " << num_arrays << ")\n";
    return ok;
}

// =====================================================================
//                       TEST GROUPS
// =====================================================================

static bool run_tiny() {
    bool ok = true;
    for (int n : {0, 1, 2, 3}) {
        std::vector<int> v = make_data(n, Pattern::Random, 1);
        ok &= testCase(v, "tiny n=" + std::to_string(n));
    }
    return ok;
}

static bool run_boundary_sizes() {
    bool ok = true;
    for (int n : {4, 5, 6, 7, 8, 9, 15, 16, 17}) {
        std::vector<int> v = make_data(n, Pattern::Random, (uint32_t)(1 + n));
        ok &= testCase(v, "network-boundary n=" + std::to_string(n));
    }
    return ok;
}

static bool run_pow2_pm1() {
    bool ok = true;
    for (int k = 4; k <= 14; ++k) {
        const int base = 1 << k;
        for (int n : {base - 1, base, base + 1}) {
            if (n < 1) continue;
            std::vector<int> v = make_data(n, Pattern::Random,
                                           (uint32_t)(n * 7919u));
            ok &= testCase(v, "pow2pm1 n=" + std::to_string(n));
        }
    }
    return ok;
}

static bool run_tiered_boundaries() {
    bool ok = true;
    for (int n : {511, 512, 513,
                  2047, 2048, 2049,
                  8191, 8192, 8193,
                  32767, 32768, 32769}) {
        std::vector<int> v = make_data(n, Pattern::Random, (uint32_t)(n * 31u));
        ok &= testCase(v, "tiered n=" + std::to_string(n),
                       ptms::ThreadMode::Auto);
    }
    return ok;
}

static bool run_all_thread_modes() {
    bool ok = true;
    const int n = 100000;
    std::vector<int> v = make_data(n, Pattern::Random, 12345);

    ok &= testCase(v, "mode Single",  ptms::ThreadMode::Single);
    ok &= testCase(v, "mode ByCores", ptms::ThreadMode::ByCores);
    ok &= testCase(v, "mode Auto",    ptms::ThreadMode::Auto);

    for (int P = 1; P <= 16; ++P) {
        ok &= testCase(v, "mode Fixed(" + std::to_string(P) + ")",
                       ptms::ThreadMode::Fixed, P);
    }
    return ok;
}

static bool run_all_thread_modes_on_odd_n() {
    bool ok = true;
    const int n = 100001;
    std::vector<int> v = make_data(n, Pattern::Random, 54321);

    for (int P : {3, 5, 6, 7, 12}) {
        ok &= testCase(v, "odd-n mode Fixed(" + std::to_string(P) + ")",
                       ptms::ThreadMode::Fixed, P);
    }
    return ok;
}

static bool run_data_patterns() {
    bool ok = true;
    const int sizes[] = {1000, 16384, 65536};

    for (int n : sizes) {
        for (Pattern p : {Pattern::Random,
                          Pattern::Sorted,
                          Pattern::Reverse,
                          Pattern::AllEqual,
                          Pattern::AlmostSorted,
                          Pattern::Extremes,
                          Pattern::TwoValues}) {
            std::vector<int> v = make_data(n, p, (uint32_t)(n + (int)p));
            std::string name = std::string("pattern ") + pattern_name(p)
                             + " n=" + std::to_string(n);
            ok &= testCase(v, name, ptms::ThreadMode::Auto);
        }
    }
    return ok;
}

static bool run_batch_wide() {
    bool ok = true;

    // Various n (hitting insertion, 2-thread, 4-thread, 8-thread paths)
    // x various num_arrays, Auto mode.
    for (int n : {4, 128, 1024, 4096, 16384}) {
        for (int arrays : {1, 10, 100}) {
            ok &= testBatch("batch Auto", n, arrays,
                            ptms::ThreadMode::Auto);
        }
    }

    // Batch with specific thread modes.
    for (int P : {1, 2, 4, 8, 16}) {
        ok &= testBatch("batch Fixed(" + std::to_string(P) + ")",
                        4096, 20, ptms::ThreadMode::Fixed, P);
    }

    // Batch with Single.
    ok &= testBatch("batch Single", 1024, 50, ptms::ThreadMode::Single);

    // Batch of size n=1 (edge case).
    ok &= testBatch("batch n=1", 1, 20, ptms::ThreadMode::Auto);

    return ok;
}

static bool run_buffer_reuse() {
    const int n = 5000;
    std::vector<int> a(n), buf(n);
    std::mt19937 rng(0xCAFEu);

    for (int trial = 0; trial < 50; ++trial) {
        std::uniform_int_distribution<int> d(-50000, 50000);
        for (auto& x : a) x = d(rng);
        std::vector<int> expected = a;
        std::sort(expected.begin(), expected.end());

        ptms::sort_int(a.data(), buf.data(), n, ptms::ThreadMode::Auto);

        if (a != expected) {
            std::cout << "[ FAIL ] buffer-reuse trial " << trial << "\n";
            return false;
        }
    }
    std::cout << "[ OK ] buffer-reuse 50 trials (n = " << n << ")\n";
    return true;
}

static bool run_idempotent_on_sorted() {
    const int n = 10000;
    std::vector<int> a = make_data(n, Pattern::Sorted, 1);
    std::vector<int> buf(n);

    ptms::sort_int(a.data(), buf.data(), n, ptms::ThreadMode::Auto);
    std::vector<int> first = a;
    ptms::sort_int(a.data(), buf.data(), n, ptms::ThreadMode::Auto);

    bool ok = (a == first);
    std::cout << "[ " << (ok ? "OK" : "FAIL")
              << " ] idempotent (sorted input)\n";
    return ok;
}

// =====================================================================
//                       MAIN
// =====================================================================

int main() {
    bool all_ok = true;

    std::cout << "=== PTMS correctness tests ===\n";
    std::cout << "SIMD level     : " << ptms::simd_level() << "\n";
    std::cout << "Physical cores : " << ptms::physical_core_count() << "\n\n";

    std::cout << "--- Tiny sizes (n < 4) ---\n";
    all_ok &= run_tiny();

    std::cout << "\n--- Network boundaries ---\n";
    all_ok &= run_boundary_sizes();

    std::cout << "\n--- Powers of two +/- 1 ---\n";
    all_ok &= run_pow2_pm1();

    std::cout << "\n--- Tiered Auto boundaries ---\n";
    all_ok &= run_tiered_boundaries();

    std::cout << "\n--- All ThreadMode values ---\n";
    all_ok &= run_all_thread_modes();

    std::cout << "\n--- Odd n with various P ---\n";
    all_ok &= run_all_thread_modes_on_odd_n();

    std::cout << "\n--- Data patterns ---\n";
    all_ok &= run_data_patterns();

    std::cout << "\n--- Batch ---\n";
    all_ok &= run_batch_wide();

    std::cout << "\n--- Buffer reuse ---\n";
    all_ok &= run_buffer_reuse();

    std::cout << "\n--- Idempotency ---\n";
    all_ok &= run_idempotent_on_sorted();

    std::cout << "\n"
              << (all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << "\n";
    return all_ok ? 0 : 1;
}
