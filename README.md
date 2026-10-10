# PTMS: Parallel Tournament Merge Sort

High-performance parallel merge sort with SIMD acceleration
(AVX2 / AVX-512) and OpenMP tasks.

## Overview

PTMS is a parallel sorting algorithm built as a three-tier pipeline:

1. **Tier 1 — Tournament merge:** 2P sorted parts are merged pairwise
   level by level (2P → P → P/2 → ... → 1). Each level runs as
   parallel OpenMP tasks.
2. **Tier 2 — Parallel sort of P chunks:** the array is split into P
   equal chunks, each sorted independently via recursive merge sort
   with ping-pong buffers.
3. **Tier 3 — Base cases:** sorting networks for blocks of 4–7
   elements; SIMD bitonic sort for blocks of 8 and 16 (AVX2 / AVX-512).

Merges inside tasks use **parallel merge with binary splitting**,
which removes the Amdahl bottleneck on the final merge.

## Performance

Benchmark on 4 million `int` (16 MB, cache-resident) on a 16-thread
AMD Ryzen 9 6900HX:

| Algorithm              | Median (s) | Speedup vs `std::sort` |
|------------------------|-----------:|----------------------:|
| `std::sort` (1 thread) |      0.300 |                 1.00x |
| `tbb::parallel_sort`   |      0.036 |                 8.30x |
| **PTMS**               |  **0.0163** |            **~18.4x** |

On AMD Ryzen 7 7500F with **AVX-512**: median **0.0162 s** at 12 threads.

## Requirements

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+ with clang-cl)
- OpenMP 4.5+
- CMake 3.15+

### Note on MSVC

MSVC does not support `#pragma omp taskgroup` even with `/openmp:llvm`.
Use **clang-cl** (bundled with Visual Studio).

## Building

```bash
git clone https://github.com/TylerBurger777/PTMS.git
cd PTMS
mkdir build && cd build
cmake ..
cmake --build . --config Release -j
```

### Manual SIMD override

```bash
cmake .. -DSIMD=AVX512   # force AVX-512
cmake .. -DSIMD=AVX2     # force AVX2
cmake .. -DSIMD=SCALAR   # disable SIMD
```

## Running the benchmark

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=true ./benchmark
```

## Running tests

```bash
cd build
ctest --output-on-failure
```

## Using the library

```cpp
#include <ptms.h>
#include <vector>

int main() {
    std::vector<int> a = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<int> buf(a.size());

    ptms::sort_int(a.data(), buf.data(), (int)a.size());
    // a = {1, 1, 2, 3, 4, 5, 6, 9}
    return 0;
}
```

## License

MIT — see [LICENSE](LICENSE).
