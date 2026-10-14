// PTMS: benchmark
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

#include "ptms.h"

#include <omp.h>

#ifdef HAVE_TBB
#include <tbb/parallel_sort.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// =====================================================================
//                       TUNING CONSTANTS
// =====================================================================

static const size_t SKIP_STD_SORT_N = 8388608;
static constexpr int    BATCH_SIZE       = 5;
static constexpr size_t COOLDOWN_MIN_N   = 1048576;
static constexpr int    COOLDOWN_MS      = 1000;
static constexpr double MAD_THRESHOLD    = 3.5;

// =====================================================================
//                       FILL HELPERS
// =====================================================================

static void fill_random_int32(int* a, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(-2000000000, 2000000000);
    for (size_t i = 0; i < n; ++i) a[i] = d(rng);
}

static void fill_random_int64(int64_t* a, size_t n, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> d(
        std::numeric_limits<int64_t>::min() / 2,
        std::numeric_limits<int64_t>::max() / 2);
    for (size_t i = 0; i < n; ++i) a[i] = d(rng);
}

static void fill_random_float(float* a, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1e6f, 1e6f);
    for (size_t i = 0; i < n; ++i) a[i] = d(rng);
}

static void fill_random_double(double* a, size_t n, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1e12, 1e12);
    for (size_t i = 0; i < n; ++i) a[i] = d(rng);
}

static void fill_random_str(std::string* a, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        int len = 8 + (int)(rng() % 9);
        std::string s(len, 'a');
        for (int k = 0; k < len; ++k) s[k] = (char)('a' + (rng() % 26));
        a[i] = std::move(s);
    }
}

// Almost-sorted: start from fully sorted, then permute `pct`% positions.
template <typename T>
static void fill_almost_sorted(T* a, size_t n, uint32_t seed, int pct) {
    for (size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i);
    std::mt19937 rng(seed);
    const size_t swaps = std::max<size_t>(1, n * (size_t)pct / 100);
    for (size_t s = 0; s < swaps; ++s) {
        size_t i = rng() % n;
        size_t j = rng() % n;
        std::swap(a[i], a[j]);
    }
}

static int runs_for(size_t n, bool heavy = false) {
    if (heavy) {
        if (n <= 65536)    return 12;
        if (n <= 262144)   return 10;
        if (n <= 1048576)  return 7;
        return 5;
    }
    if (n <= 4096)     return 50;
    if (n <= 65536)    return 30;
    if (n <= 524288)   return 20;
    if (n <= 2097152)  return 12;
    return 8;
}

// =====================================================================
//                       OUTLIER FILTER (MAD)
// =====================================================================

static std::vector<double> remove_outliers_mad(const std::vector<double>& sorted_v) {
    const size_t n = sorted_v.size();
    if (n < 8) return sorted_v;

    const double med = sorted_v[n / 2];

    std::vector<double> absdev(n);
    for (size_t i = 0; i < n; ++i) absdev[i] = std::fabs(sorted_v[i] - med);
    std::sort(absdev.begin(), absdev.end());
    const double mad = absdev[n / 2];

    if (mad <= 0.0) return sorted_v;

    const double scale = 0.6745 / mad;

    std::vector<double> filtered;
    filtered.reserve(n);
    for (double x : sorted_v) {
        const double z = scale * (x - med);
        if (std::fabs(z) <= MAD_THRESHOLD) filtered.push_back(x);
    }
    if (filtered.empty()) return sorted_v;
    return filtered;
}

// =====================================================================
//                       MEASUREMENT
// =====================================================================

struct AlgoResult {
    std::string name;
    double median_ms = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double melem_s   = 0.0;
    size_t kept      = 0;
    size_t total     = 0;
};

struct InterleavedAlgo {
    std::string name;
    std::function<void()> copy;
    std::function<void()> run;
};

static std::vector<AlgoResult> run_interleaved(
    const std::vector<InterleavedAlgo>& algos,
    int total_runs, size_t N)
{
    const size_t A = algos.size();
    std::vector<std::vector<double>> times(A);
    for (auto& t : times) t.reserve(total_runs);

    const int WARMUP = (N <= 1048576) ? 5 : 3;
    for (int w = 0; w < WARMUP; ++w) {
        for (size_t a = 0; a < A; ++a) {
            algos[a].copy();
            algos[a].run();
        }
    }

    const bool need_cooldown = (N >= COOLDOWN_MIN_N);

    int remaining = total_runs;
    while (remaining > 0) {
        const int this_batch = std::min(BATCH_SIZE, remaining);
        for (int r = 0; r < this_batch; ++r) {
            for (size_t a = 0; a < A; ++a) {
                algos[a].copy();
                auto t0 = Clock::now();
                algos[a].run();
                auto t1 = Clock::now();
                times[a].push_back(Sec(t1 - t0).count() * 1000.0);
            }
        }
        remaining -= this_batch;
        if (need_cooldown && remaining > 0) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(COOLDOWN_MS));
        }
    }

    std::vector<AlgoResult> results;
    results.reserve(A);
    for (size_t a = 0; a < A; ++a) {
        auto& v = times[a];
        std::sort(v.begin(), v.end());
        const size_t total = v.size();
        std::vector<double> kept = remove_outliers_mad(v);

        AlgoResult r;
        r.name      = algos[a].name;
        r.total     = total;
        r.kept      = kept.size();
        r.min_ms    = kept.front();
        r.max_ms    = kept.back();
        r.median_ms = kept[kept.size() / 2];
        r.melem_s   = (r.min_ms > 0.0)
                    ? (double)N / r.min_ms / 1000.0
                    : 0.0;
        results.push_back(r);
    }
    return results;
}

static void print_header(const char* section, const char* baseline_note) {
    std::printf("\n========== %s ==========\n", section);
    std::printf("Interleaved + batched. MAD filter |z| <= %.1f.\n", MAD_THRESHOLD);
    std::printf("%s\n", baseline_note);
    std::printf("%-9s  %-22s %10s %10s %10s %9s %9s  %8s %s\n",
                "N", "Algorithm",
                "Median", "Min", "Max", "Speedup", "Melem/s",
                "Jitter", "Kept");
    std::printf("%s\n", std::string(104, '-').c_str());
    std::printf("Kept = samples after MAD filter / total. Jitter = (Max-Min)/Median.\n\n");
}

static void print_row(size_t N, const AlgoResult& r, double base_ms) {
    double sp     = (base_ms > 0.0) ? base_ms / r.median_ms : 0.0;
    double jitter = (r.median_ms > 0.0)
                  ? 100.0 * (r.max_ms - r.min_ms) / r.median_ms
                  : 0.0;
    char kept[24];
    std::snprintf(kept, sizeof(kept), "%zu/%zu", r.kept, r.total);
    std::printf("%-9zu  %-22s %10.3f %10.3f %10.3f %8.2fx %9.1f  %7.1f%% %s\n",
                N, r.name.c_str(),
                r.median_ms, r.min_ms, r.max_ms,
                sp, r.melem_s, jitter, kept);
}

static void print_winner(const std::vector<AlgoResult>& results, double base_min_ms,
                         const char* baseline_name) {
    if (results.empty()) return;
    const AlgoResult* best = &results.front();
    for (const auto& r : results)
        if (r.min_ms < best->min_ms) best = &r;
    double sp = (base_min_ms > 0.0) ? base_min_ms / best->min_ms : 0.0;
    std::printf(">>> Winner: %s (%.3f ms, %.2fx vs %s)\n\n",
                best->name.c_str(), best->min_ms, sp, baseline_name);
}

// =====================================================================
//                       RADIX (int32)
// =====================================================================

static void radix_sort_int32_serial(int* a, int n) {
    if (n <= 1) return;
    std::vector<int> buf(n);
    int* src = a; int* dst = buf.data();
    for (int shift = 0; shift < 32; shift += 8) {
        const bool last = (shift == 24);
        int cnt[256] = {0};
        for (int i = 0; i < n; ++i) {
            int key = (src[i] >> shift) & 0xFF;
            if (last) key ^= 0x80;
            ++cnt[key];
        }
        int sum = 0;
        for (int i = 0; i < 256; ++i) { int c = cnt[i]; cnt[i] = sum; sum += c; }
        for (int i = 0; i < n; ++i) {
            int key = (src[i] >> shift) & 0xFF;
            if (last) key ^= 0x80;
            dst[cnt[key]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, (size_t)n * sizeof(int));
}

static void radix_sort_int32_parallel(int* a, int* buf, int n) {
    int P = omp_get_max_threads();
    if (n < 65536 || P <= 1) { radix_sort_int32_serial(a, n); return; }
    int* src = a; int* dst = buf;
    std::vector<int> hist(P * 256), offsets(P * 256);
    for (int shift = 0; shift < 32; shift += 8) {
        const bool last = (shift == 24);
        #pragma omp parallel num_threads(P)
        {
            int t = omp_get_thread_num();
            int* h = hist.data() + t * 256;
            std::memset(h, 0, 256 * sizeof(int));
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                int key = (src[i] >> shift) & 0xFF;
                if (last) key ^= 0x80;
                ++h[key];
            }
            #pragma omp single
            {
                int bt[256] = {0};
                for (int tt = 0; tt < P; ++tt) {
                    const int* hh = hist.data() + tt * 256;
                    for (int b = 0; b < 256; ++b) bt[b] += hh[b];
                }
                int run = 0;
                for (int b = 0; b < 256; ++b) {
                    int inner = run;
                    run += bt[b];
                    for (int tt = 0; tt < P; ++tt) {
                        offsets[tt * 256 + b] = inner;
                        inner += hist[tt * 256 + b];
                    }
                }
            }
            int lo = (int)((long long)t * n / P);
            int hi = (int)((long long)(t + 1) * n / P);
            int* off = offsets.data() + t * 256;
            for (int i = lo; i < hi; ++i) {
                int key = (src[i] >> shift) & 0xFF;
                if (last) key ^= 0x80;
                dst[off[key]++] = src[i];
            }
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, (size_t)n * sizeof(int));
}

// =====================================================================
//                       RADIX (int64)
// =====================================================================

static void radix_sort_int64_serial(int64_t* a, int n) {
    if (n <= 1) return;
    std::vector<int64_t> buf(n);
    int64_t* src = a; int64_t* dst = buf.data();
    for (int shift = 0; shift < 64; shift += 8) {
        const bool last = (shift == 56);
        int cnt[256] = {0};
        for (int i = 0; i < n; ++i) {
            int key = (int)((src[i] >> shift) & 0xFF);
            if (last) key ^= 0x80;
            ++cnt[key];
        }
        int sum = 0;
        for (int i = 0; i < 256; ++i) { int c = cnt[i]; cnt[i] = sum; sum += c; }
        for (int i = 0; i < n; ++i) {
            int key = (int)((src[i] >> shift) & 0xFF);
            if (last) key ^= 0x80;
            dst[cnt[key]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, (size_t)n * sizeof(int64_t));
}

static void radix_sort_int64_parallel(int64_t* a, int64_t* buf, int n) {
    int P = omp_get_max_threads();
    if (n < 65536 || P <= 1) { radix_sort_int64_serial(a, n); return; }
    int64_t* src = a; int64_t* dst = buf;
    std::vector<int> hist(P * 256), offsets(P * 256);
    for (int shift = 0; shift < 64; shift += 8) {
        const bool last = (shift == 56);
        #pragma omp parallel num_threads(P)
        {
            int t = omp_get_thread_num();
            int* h = hist.data() + t * 256;
            std::memset(h, 0, 256 * sizeof(int));
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                int key = (int)((src[i] >> shift) & 0xFF);
                if (last) key ^= 0x80;
                ++h[key];
            }
            #pragma omp single
            {
                int bt[256] = {0};
                for (int tt = 0; tt < P; ++tt) {
                    const int* hh = hist.data() + tt * 256;
                    for (int b = 0; b < 256; ++b) bt[b] += hh[b];
                }
                int run = 0;
                for (int b = 0; b < 256; ++b) {
                    int inner = run;
                    run += bt[b];
                    for (int tt = 0; tt < P; ++tt) {
                        offsets[tt * 256 + b] = inner;
                        inner += hist[tt * 256 + b];
                    }
                }
            }
            int lo = (int)((long long)t * n / P);
            int hi = (int)((long long)(t + 1) * n / P);
            int* off = offsets.data() + t * 256;
            for (int i = lo; i < hi; ++i) {
                int key = (int)((src[i] >> shift) & 0xFF);
                if (last) key ^= 0x80;
                dst[off[key]++] = src[i];
            }
        }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, (size_t)n * sizeof(int64_t));
}

// =====================================================================
//                       RADIX (float)
// =====================================================================

static inline uint32_t float_to_sortable(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float sortable_to_float(uint32_t u) {
    u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    float f; std::memcpy(&f, &u, 4); return f;
}

static void radix_sort_float_serial(float* a, int n) {
    if (n <= 1) return;
    std::vector<uint32_t> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = float_to_sortable(a[i]);
    std::vector<uint32_t> buf(n);
    uint32_t* src = keys.data(); uint32_t* dst = buf.data();
    for (int shift = 0; shift < 32; shift += 8) {
        int cnt[256] = {0};
        for (int i = 0; i < n; ++i) ++cnt[(src[i] >> shift) & 0xFF];
        int sum = 0;
        for (int i = 0; i < 256; ++i) { int c = cnt[i]; cnt[i] = sum; sum += c; }
        for (int i = 0; i < n; ++i) dst[cnt[(src[i] >> shift) & 0xFF]++] = src[i];
        std::swap(src, dst);
    }
    for (int i = 0; i < n; ++i) a[i] = sortable_to_float(src[i]);
}

static void radix_sort_float_parallel(float* a, uint32_t* buf, int n) {
    int P = omp_get_max_threads();
    if (n < 65536 || P <= 1) { radix_sort_float_serial(a, n); return; }
    for (int i = 0; i < n; ++i) buf[i] = float_to_sortable(a[i]);
    uint32_t* src = buf;
    uint32_t* dst = reinterpret_cast<uint32_t*>(a);
    std::vector<int> hist(P * 256), offsets(P * 256);
    for (int shift = 0; shift < 32; shift += 8) {
        #pragma omp parallel num_threads(P)
        {
            int t = omp_get_thread_num();
            int* h = hist.data() + t * 256;
            std::memset(h, 0, 256 * sizeof(int));
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) ++h[(src[i] >> shift) & 0xFF];
            #pragma omp single
            {
                int bt[256] = {0};
                for (int tt = 0; tt < P; ++tt) {
                    const int* hh = hist.data() + tt * 256;
                    for (int b = 0; b < 256; ++b) bt[b] += hh[b];
                }
                int run = 0;
                for (int b = 0; b < 256; ++b) {
                    int inner = run;
                    run += bt[b];
                    for (int tt = 0; tt < P; ++tt) {
                        offsets[tt * 256 + b] = inner;
                        inner += hist[tt * 256 + b];
                    }
                }
            }
            int lo = (int)((long long)t * n / P);
            int hi = (int)((long long)(t + 1) * n / P);
            int* off = offsets.data() + t * 256;
            for (int i = lo; i < hi; ++i) {
                int k = (src[i] >> shift) & 0xFF;
                dst[off[k]++] = src[i];
            }
        }
        std::swap(src, dst);
    }
    for (int i = 0; i < n; ++i) a[i] = sortable_to_float(src[i]);
}

// =====================================================================
//                       RADIX (double)
// =====================================================================

static inline uint64_t double_to_sortable(double d) {
    uint64_t u; std::memcpy(&u, &d, 8);
    return (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
}
static inline double sortable_to_double(uint64_t u) {
    u = (u & 0x8000000000000000ull) ? (u & 0x7FFFFFFFFFFFFFFFull) : ~u;
    double d; std::memcpy(&d, &u, 8); return d;
}

static void radix_sort_double_serial(double* a, int n) {
    if (n <= 1) return;
    std::vector<uint64_t> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = double_to_sortable(a[i]);
    std::vector<uint64_t> buf(n);
    uint64_t* src = keys.data(); uint64_t* dst = buf.data();
    for (int shift = 0; shift < 64; shift += 8) {
        int cnt[256] = {0};
        for (int i = 0; i < n; ++i) ++cnt[(src[i] >> shift) & 0xFF];
        int sum = 0;
        for (int i = 0; i < 256; ++i) { int c = cnt[i]; cnt[i] = sum; sum += c; }
        for (int i = 0; i < n; ++i) dst[cnt[(src[i] >> shift) & 0xFF]++] = src[i];
        std::swap(src, dst);
    }
    for (int i = 0; i < n; ++i) a[i] = sortable_to_double(src[i]);
}

static void radix_sort_double_parallel(double* a, uint64_t* buf, int n) {
    int P = omp_get_max_threads();
    if (n < 65536 || P <= 1) { radix_sort_double_serial(a, n); return; }
    for (int i = 0; i < n; ++i) buf[i] = double_to_sortable(a[i]);
    uint64_t* src = buf;
    uint64_t* dst = reinterpret_cast<uint64_t*>(a);
    std::vector<int> hist(P * 256), offsets(P * 256);
    for (int shift = 0; shift < 64; shift += 8) {
        #pragma omp parallel num_threads(P)
        {
            int t = omp_get_thread_num();
            int* h = hist.data() + t * 256;
            std::memset(h, 0, 256 * sizeof(int));
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) ++h[(src[i] >> shift) & 0xFF];
            #pragma omp single
            {
                int bt[256] = {0};
                for (int tt = 0; tt < P; ++tt) {
                    const int* hh = hist.data() + tt * 256;
                    for (int b = 0; b < 256; ++b) bt[b] += hh[b];
                }
                int run = 0;
                for (int b = 0; b < 256; ++b) {
                    int inner = run;
                    run += bt[b];
                    for (int tt = 0; tt < P; ++tt) {
                        offsets[tt * 256 + b] = inner;
                        inner += hist[tt * 256 + b];
                    }
                }
            }
            int lo = (int)((long long)t * n / P);
            int hi = (int)((long long)(t + 1) * n / P);
            int* off = offsets.data() + t * 256;
            for (int i = lo; i < hi; ++i) {
                int k = (int)((src[i] >> shift) & 0xFF);
                dst[off[k]++] = src[i];
            }
        }
        std::swap(src, dst);
    }
    for (int i = 0; i < n; ++i) a[i] = sortable_to_double(src[i]);
}

// =====================================================================
//                       SELF-TESTS / VERIFICATION
// =====================================================================

static bool self_test_network(int n) {
    std::vector<int> a(n), buf(n);
    const int TRIALS = 2000;
    for (int t = 0; t < TRIALS; ++t) {
        std::mt19937 rng((uint32_t)(t * 7919 + n));
        std::uniform_int_distribution<int> d(-10000, 10000);
        for (auto& x : a) x = d(rng);
        std::vector<int> expected = a;
        std::sort(expected.begin(), expected.end());
        ptms::sort_int(a.data(), buf.data(), n, ptms::ThreadMode::Single, 0);
        if (a != expected) return false;
    }
    return true;
}

static void run_self_tests() {
    for (int n = 4; n <= 7; ++n) {
        bool ok = self_test_network(n);
        std::printf("Self-test sort %d network : %s\n", n, ok ? "PASS" : "FAIL");
    }
    std::printf("\n");
}

static void verify_algorithms() {
    std::printf("========== Algorithm verification (N = 100000) ==========\n");
    const int N = 100000;

    {
        std::vector<int> original(N);
        fill_random_int32(original.data(), N, 0xABCDu);
        std::vector<int> ref = original;
        std::sort(ref.begin(), ref.end());
        std::vector<int> a, b(N);
        a = original; std::sort(a.begin(), a.end());
        std::printf("std::sort          (int32)   : %s\n", a == ref ? "OK" : "FAIL");
#ifdef HAVE_TBB
        a = original; tbb::parallel_sort(a.begin(), a.end());
        std::printf("tbb::parallel_sort (int32)   : %s\n", a == ref ? "OK" : "FAIL");
#endif
        a = original; ptms::sort_int(a.data(), b.data(), N, ptms::ThreadMode::Auto);
        std::printf("ptms::sort_int     (int32)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_int32_serial(a.data(), N);
        std::printf("radix_serial       (int32)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_int32_parallel(a.data(), b.data(), N);
        std::printf("radix_parallel     (int32)   : %s\n", a == ref ? "OK" : "FAIL");
    }
    {
        std::vector<int64_t> original(N);
        fill_random_int64(original.data(), N, 0x1234u);
        std::vector<int64_t> ref = original;
        std::sort(ref.begin(), ref.end());
        std::vector<int64_t> a, b(N);
        a = original; std::sort(a.begin(), a.end());
        std::printf("std::sort          (int64)   : %s\n", a == ref ? "OK" : "FAIL");
#ifdef HAVE_TBB
        a = original; tbb::parallel_sort(a.begin(), a.end());
        std::printf("tbb::parallel_sort (int64)   : %s\n", a == ref ? "OK" : "FAIL");
#endif
        a = original; ptms::sort<int64_t>(a.data(), b.data(), N);
        std::printf("ptms::sort<int64>  (int64)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_int64_serial(a.data(), N);
        std::printf("radix_serial       (int64)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_int64_parallel(a.data(), b.data(), N);
        std::printf("radix_parallel     (int64)   : %s\n", a == ref ? "OK" : "FAIL");
    }
    {
        std::vector<float> original(N);
        fill_random_float(original.data(), N, 0x5678u);
        std::vector<float> ref = original;
        std::sort(ref.begin(), ref.end());
        std::vector<float> a, b(N);
        std::vector<uint32_t> ubuf(N);
        a = original; std::sort(a.begin(), a.end());
        std::printf("std::sort          (float)   : %s\n", a == ref ? "OK" : "FAIL");
#ifdef HAVE_TBB
        a = original; tbb::parallel_sort(a.begin(), a.end());
        std::printf("tbb::parallel_sort (float)   : %s\n", a == ref ? "OK" : "FAIL");
#endif
        a = original; ptms::sort<float>(a.data(), b.data(), N);
        std::printf("ptms::sort<float>  (float)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_float_serial(a.data(), N);
        std::printf("radix_serial       (float)   : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_float_parallel(a.data(), ubuf.data(), N);
        std::printf("radix_parallel     (float)   : %s\n", a == ref ? "OK" : "FAIL");
    }
    {
        std::vector<double> original(N);
        fill_random_double(original.data(), N, 0x9ABCu);
        std::vector<double> ref = original;
        std::sort(ref.begin(), ref.end());
        std::vector<double> a, b(N);
        std::vector<uint64_t> ubuf(N);
        a = original; std::sort(a.begin(), a.end());
        std::printf("std::sort          (double)  : %s\n", a == ref ? "OK" : "FAIL");
#ifdef HAVE_TBB
        a = original; tbb::parallel_sort(a.begin(), a.end());
        std::printf("tbb::parallel_sort (double)  : %s\n", a == ref ? "OK" : "FAIL");
#endif
        a = original; ptms::sort<double>(a.data(), b.data(), N);
        std::printf("ptms::sort<double> (double)  : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_double_serial(a.data(), N);
        std::printf("radix_serial       (double)  : %s\n", a == ref ? "OK" : "FAIL");
        a = original; radix_sort_double_parallel(a.data(), ubuf.data(), N);
        std::printf("radix_parallel     (double)  : %s\n", a == ref ? "OK" : "FAIL");
    }
    {
        std::vector<std::string> original(N);
        fill_random_str(original.data(), N, 0xDEF0u);
        std::vector<std::string> ref = original;
        std::sort(ref.begin(), ref.end());
        std::vector<std::string> a = original, b(N);
        a = original; std::sort(a.begin(), a.end());
        std::printf("std::sort          (string)  : %s\n", a == ref ? "OK" : "FAIL");
#ifdef HAVE_TBB
        a = original; tbb::parallel_sort(a.begin(), a.end());
        std::printf("tbb::parallel_sort (string)  : %s\n", a == ref ? "OK" : "FAIL");
#endif
        a = original; ptms::sort<std::string>(a.data(), b.data(), N);
        std::printf("ptms::sort<string> (string)  : %s\n", a == ref ? "OK" : "FAIL");
    }

    std::printf("\n");
}

// =====================================================================
//                       SIZES
// =====================================================================

static const size_t kSizesNum[] = {
    1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152,
    4194304, 8388608, 16777216
};

static const size_t kSizesStr[] = {
    1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152,
    4194304
};

// =====================================================================
//                       BENCH: INT32
// =====================================================================

static void bench_int32() {
    print_header("int32",
        "std::sort measured for N < 8M; above that baseline is tbb.");

    for (size_t N : kSizesNum) {
        std::vector<int> original(N);
        fill_random_int32(original.data(), N, (uint32_t)N);

        std::vector<int> w_std(N), w_tbb(N), w_ptms(N), w_rs(N), w_rp(N);
        std::vector<int> scratch_ptms(N), scratch_rp(N);
        int* bp = scratch_ptms.data();
        int* br = scratch_rp.data();

        std::vector<InterleavedAlgo> algos;

        const bool measure_std = (N < SKIP_STD_SORT_N);
        if (measure_std) {
            algos.push_back({
                "std::sort",
                [&]() { std::copy(original.begin(), original.end(), w_std.begin()); },
                [&]() { std::sort(w_std.data(), w_std.data() + N); }
            });
        }
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort_int",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort_int(w_ptms.data(), bp, (int)N, ptms::ThreadMode::Auto); }
        });
        algos.push_back({
            "radix_serial",
            [&]() { std::copy(original.begin(), original.end(), w_rs.begin()); },
            [&]() { radix_sort_int32_serial(w_rs.data(), (int)N); }
        });
        algos.push_back({
            "radix_parallel",
            [&]() { std::copy(original.begin(), original.end(), w_rp.begin()); },
            [&]() { radix_sort_int32_parallel(w_rp.data(), br, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       BENCH: INT64
// =====================================================================

static void bench_int64() {
    print_header("int64",
        "Radix does 8 passes. std::sort measured for N < 8M.");

    for (size_t N : kSizesNum) {
        std::vector<int64_t> original(N);
        fill_random_int64(original.data(), N, (uint32_t)N);

        std::vector<int64_t> w_std(N), w_tbb(N), w_ptms(N), w_rs(N), w_rp(N);
        std::vector<int64_t> scratch_ptms(N), scratch_rp(N);
        int64_t* bp = scratch_ptms.data();
        int64_t* br = scratch_rp.data();

        std::vector<InterleavedAlgo> algos;

        const bool measure_std = (N < SKIP_STD_SORT_N);
        if (measure_std) {
            algos.push_back({
                "std::sort",
                [&]() { std::copy(original.begin(), original.end(), w_std.begin()); },
                [&]() { std::sort(w_std.data(), w_std.data() + N); }
            });
        }
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<int64>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<int64_t>(w_ptms.data(), bp, (int)N); }
        });
        algos.push_back({
            "radix_serial",
            [&]() { std::copy(original.begin(), original.end(), w_rs.begin()); },
            [&]() { radix_sort_int64_serial(w_rs.data(), (int)N); }
        });
        algos.push_back({
            "radix_parallel",
            [&]() { std::copy(original.begin(), original.end(), w_rp.begin()); },
            [&]() { radix_sort_int64_parallel(w_rp.data(), br, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       BENCH: FLOAT
// =====================================================================

static void bench_float() {
    print_header("float",
        "std::sort measured for N < 8M; above that baseline is tbb.");

    for (size_t N : kSizesNum) {
        std::vector<float> original(N);
        fill_random_float(original.data(), N, (uint32_t)N);

        std::vector<float> w_std(N), w_tbb(N), w_ptms(N), w_rs(N), w_rp(N);
        std::vector<float> scratch_ptms(N);
        std::vector<uint32_t> scratch_rp(N);
        float*    bp = scratch_ptms.data();
        uint32_t* br = scratch_rp.data();

        std::vector<InterleavedAlgo> algos;

        const bool measure_std = (N < SKIP_STD_SORT_N);
        if (measure_std) {
            algos.push_back({
                "std::sort",
                [&]() { std::copy(original.begin(), original.end(), w_std.begin()); },
                [&]() { std::sort(w_std.data(), w_std.data() + N); }
            });
        }
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<float>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<float>(w_ptms.data(), bp, (int)N); }
        });
        algos.push_back({
            "radix_serial",
            [&]() { std::copy(original.begin(), original.end(), w_rs.begin()); },
            [&]() { radix_sort_float_serial(w_rs.data(), (int)N); }
        });
        algos.push_back({
            "radix_parallel",
            [&]() { std::copy(original.begin(), original.end(), w_rp.begin()); },
            [&]() { radix_sort_float_parallel(w_rp.data(), br, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       BENCH: DOUBLE
// =====================================================================

static void bench_double() {
    print_header("double",
        "Radix does 8 passes. std::sort measured for N < 8M.");

    for (size_t N : kSizesNum) {
        std::vector<double> original(N);
        fill_random_double(original.data(), N, (uint32_t)N);

        std::vector<double> w_std(N), w_tbb(N), w_ptms(N), w_rs(N), w_rp(N);
        std::vector<double> scratch_ptms(N);
        std::vector<uint64_t> scratch_rp(N);
        double*   bp = scratch_ptms.data();
        uint64_t* br = scratch_rp.data();

        std::vector<InterleavedAlgo> algos;

        const bool measure_std = (N < SKIP_STD_SORT_N);
        if (measure_std) {
            algos.push_back({
                "std::sort",
                [&]() { std::copy(original.begin(), original.end(), w_std.begin()); },
                [&]() { std::sort(w_std.data(), w_std.data() + N); }
            });
        }
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<double>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<double>(w_ptms.data(), bp, (int)N); }
        });
        algos.push_back({
            "radix_serial",
            [&]() { std::copy(original.begin(), original.end(), w_rs.begin()); },
            [&]() { radix_sort_double_serial(w_rs.data(), (int)N); }
        });
        algos.push_back({
            "radix_parallel",
            [&]() { std::copy(original.begin(), original.end(), w_rp.begin()); },
            [&]() { radix_sort_double_parallel(w_rp.data(), br, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       BENCH: STRING
// =====================================================================

static void bench_string() {
    print_header("std::string",
        "Radix not applicable. std::sort measured for N < 8M.");

    for (size_t N : kSizesStr) {
        std::vector<std::string> original(N);
        fill_random_str(original.data(), N, (uint32_t)N);

        std::vector<std::string> w_std(N), w_tbb(N), w_ptms(N);
        std::vector<std::string> scratch_ptms(N);
        std::string* bp = scratch_ptms.data();

        std::vector<InterleavedAlgo> algos;

        const bool measure_std = (N < SKIP_STD_SORT_N);
        if (measure_std) {
            algos.push_back({
                "std::sort",
                [&]() { std::copy(original.begin(), original.end(), w_std.begin()); },
                [&]() { std::sort(w_std.begin(), w_std.end()); }
            });
        }
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<string>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<std::string>(w_ptms.data(), bp, (int)N); }
        });

        int runs = runs_for(N, /*heavy=*/true);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       BENCH: ALMOST SORTED
// =====================================================================

// Tests the adaptive merge skip. Data is fully sorted, then 1% of
// positions are randomly swapped. On such input, PTMS's adaptive skip
// collapses whole levels of the tournament tree to memcpy.
static void bench_almost_sorted() {
    const size_t sizes[] = {
        65536, 262144, 1048576, 4194304, 16777216
    };
    const int kPct = 1;  // 1% of positions shuffled

    print_header("almost sorted (1% positions shuffled)",
        "Tests adaptive merge skip. Baseline: tbb::parallel_sort.");

    // ---- int32 ----
    std::printf("--- int32 ---\n");
    for (size_t N : sizes) {
        std::vector<int> original(N);
        fill_almost_sorted(original.data(), N, (uint32_t)N, kPct);

        std::vector<int> w_tbb(N), w_ptms(N);
        std::vector<int> scratch_ptms(N);
        int* bp = scratch_ptms.data();

        std::vector<InterleavedAlgo> algos;
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort_int",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort_int(w_ptms.data(), bp, (int)N, ptms::ThreadMode::Auto); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }

    // ---- int64 ----
    std::printf("--- int64 ---\n");
    for (size_t N : sizes) {
        std::vector<int64_t> original(N);
        fill_almost_sorted(original.data(), N, (uint32_t)N, kPct);

        std::vector<int64_t> w_tbb(N), w_ptms(N);
        std::vector<int64_t> scratch_ptms(N);
        int64_t* bp = scratch_ptms.data();

        std::vector<InterleavedAlgo> algos;
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<int64>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<int64_t>(w_ptms.data(), bp, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }

    // ---- double ----
    std::printf("--- double ---\n");
    for (size_t N : sizes) {
        std::vector<double> original(N);
        fill_almost_sorted(original.data(), N, (uint32_t)N, kPct);

        std::vector<double> w_tbb(N), w_ptms(N);
        std::vector<double> scratch_ptms(N);
        double* bp = scratch_ptms.data();

        std::vector<InterleavedAlgo> algos;
#ifdef HAVE_TBB
        algos.push_back({
            "tbb::parallel_sort",
            [&]() { std::copy(original.begin(), original.end(), w_tbb.begin()); },
            [&]() { tbb::parallel_sort(w_tbb.begin(), w_tbb.end()); }
        });
#endif
        algos.push_back({
            "ptms::sort<double>",
            [&]() { std::copy(original.begin(), original.end(), w_ptms.begin()); },
            [&]() { ptms::sort<double>(w_ptms.data(), bp, (int)N); }
        });

        int runs = runs_for(N);
        auto results = run_interleaved(algos, runs, N);

        const double base_ms   = results[0].median_ms;
        const double base_min  = results[0].min_ms;
        const char*  base_name = results[0].name.c_str();

        for (const auto& r : results) print_row(N, r, base_ms);
        print_winner(results, base_min, base_name);
    }
}

// =====================================================================
//                       MAIN
// =====================================================================

int main() {
    std::printf("=== PTMS: Parallel Tournament Merge Sort ===\n");
    std::printf("SIMD level     : %s\n", ptms::simd_level());
    std::printf("Physical cores : %d\n", ptms::physical_core_count());
    std::printf("Logical threads: %d\n", omp_get_max_threads());
#ifdef HAVE_TBB
    std::printf("TBB            : enabled\n");
#else
    std::printf("TBB            : not available\n");
#endif
#ifdef NDEBUG
    std::printf("Build          : RELEASE\n");
#else
    std::printf("Build          : DEBUG\n");
#endif
    std::printf("Skip std::sort : N >= %zu\n", SKIP_STD_SORT_N);
    std::printf("Batch: %d runs, cooldown %d ms for N >= %zu\n",
                BATCH_SIZE, COOLDOWN_MS, COOLDOWN_MIN_N);
    std::printf("Outlier filter : MAD, |modified z| <= %.1f\n", MAD_THRESHOLD);
    std::printf("\n");

    run_self_tests();
    verify_algorithms();

    bench_int32();
    bench_int64();
    bench_float();
    bench_double();
    //bench_string();
    bench_almost_sorted();

    return 0;
}
