// PTMS: correctness tests
//
// Copyright (c) 2025 Andrey Afimov
// SPDX-License-Identifier: MIT

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

    all_ok &= testCase({},                         "empty");
    all_ok &= testCase({42},                       "single element");
    all_ok &= testCase({1, 2},                     "two sorted");
    all_ok &= testCase({2, 1},                     "two reversed");
    all_ok &= testCase({5, 5, 5, 5, 5},            "all equal");
    all_ok &= testCase({1,2,3,4,5,6,7,8},          "already sorted");
    all_ok &= testCase({8,7,6,5,4,3,2,1},          "reversed");

    for (int n : {100, 500, 1000, 10000, 100000}) {
        std::mt19937 rng(12345 + n);
        std::uniform_int_distribution<int> d(-10000, 10000);
        std::vector<int> v(n);
        for (auto& x : v) x = d(rng);
        all_ok &= testCase(v, "random size");
    }

    {
        std::vector<int> v(100000);
        std::mt19937 rng(999);
        std::uniform_int_distribution<int> d2(0, 10);
        for (auto& x : v) x = d2(rng);
        all_ok &= testCase(v, "many duplicates");
    }

    std::cout << "\n" << (all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
