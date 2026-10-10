// PTMS: correctness tests
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT
//
// Note: tests use arrays of size >= 4, since the parallel pipeline
// is only engaged for n >= 4 (see README "Limitations" section).
// Arrays of size 0..3 are handled by an internal insertion-sort
// fallback and are not tested here.

#include "ptms.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

static bool testCase(std::vector<int> data, const char* name) {
    std::vector<int> a = data;
    std::vector<int> buf(a.size());
    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());

    ptms::sort_int(a.data(), buf.data(), (int)a.size());

    bool ok = (a == expected);
    std::cout << "[ " << (ok ? "OK" : "FAIL") << " ] " << name
              << " (n = " << a.size() << ")\n";
    return ok;
}

int main() {
    bool all_ok = true;

    // ---- Boundary cases around network sizes (4..7) ----
    for (int n : {4, 5, 6, 7, 8, 16}) {
        std::mt19937 rng(12345 + n);
        std::uniform_int_distribution<int> d(-10000, 10000);
        std::vector<int> v(n);
        for (auto& x : v) x = d(rng);
        all_ok &= testCase(v, "boundary size");
    }

    // ---- Already sorted ----
    {
        std::vector<int> v(64);
        for (int i = 0; i < 64; ++i) v[i] = i;
        all_ok &= testCase(v, "already sorted (n = 64)");
    }

    // ---- Reverse sorted ----
    {
        std::vector<int> v(64);
        for (int i = 0; i < 64; ++i) v[i] = 64 - i;
        all_ok &= testCase(v, "reversed (n = 64)");
    }

    // ---- All equal ----
    {
        std::vector<int> v(100, 7);
        all_ok &= testCase(v, "all equal (n = 100)");
    }

    // ---- Random sizes ----
    for (int n : {100, 500, 1000, 10000, 100000}) {
        std::mt19937 rng(12345 + n);
        std::uniform_int_distribution<int> d(-10000, 10000);
        std::vector<int> v(n);
        for (auto& x : v) x = d(rng);
        all_ok &= testCase(v, "random size");
    }

    // ---- Many duplicates ----
    {
        std::vector<int> v(100000);
        std::mt19937 rng(999);
        std::uniform_int_distribution<int> d2(0, 10);
        for (auto& x : v) x = d2(rng);
        all_ok &= testCase(v, "many duplicates (n = 100000)");
    }

    std::cout << "\n" << (all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
