cmake_minimum_required(VERSION 3.15)

project(ptms
    VERSION 1.0.0
    DESCRIPTION "Parallel Tournament Merge Sort with SIMD (AVX2/AVX-512)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)

# SIMD selection
set(SIMD "AUTO" CACHE STRING "SIMD level: AUTO, AVX512, AVX2, SCALAR")
set_property(CACHE SIMD PROPERTY STRINGS AUTO AVX512 AVX2 SCALAR)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

if(SIMD STREQUAL "AUTO")
    message(STATUS "SIMD: auto-detecting...")
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() { __m512i a = _mm512_setzero_si512(); (void)a; return 0; }
        " HAS_AVX512_MSVC)
        if(HAS_AVX512_MSVC)
            message(STATUS "SIMD: AVX-512 (MSVC)")
            add_compile_options(/arch:AVX512)
        else()
            message(STATUS "SIMD: AVX2 (MSVC)")
            add_compile_options(/arch:AVX2)
        endif()
    else()
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            add_compile_options(-march=native)
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -march=native -dM -E -x c++ /dev/null
                OUTPUT_VARIABLE NATIVE_DEFINES ERROR_QUIET)
            if(NATIVE_DEFINES MATCHES "__AVX512F__")
                message(STATUS "SIMD: AVX-512 (native)")
            elseif(NATIVE_DEFINES MATCHES "__AVX2__")
                message(STATUS "SIMD: AVX2 (native)")
            else()
                message(STATUS "SIMD: scalar (native)")
            endif()
        else()
            message(STATUS "SIMD: AVX2 (fallback)")
            add_compile_options(-mavx2)
        endif()
    endif()
elseif(SIMD STREQUAL "AVX512")
    message(STATUS "SIMD: AVX-512 (user override)")
    if(MSVC)
        add_compile_options(/arch:AVX512)
    else()
        add_compile_options(-mavx512f)
    endif()
elseif(SIMD STREQUAL "AVX2")
    message(STATUS "SIMD: AVX2 (user override)")
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2)
    endif()
elseif(SIMD STREQUAL "SCALAR")
    message(STATUS "SIMD: scalar (user override)")
else()
    message(FATAL_ERROR "Unknown SIMD mode: ${SIMD}")
endif()

if(MSVC)
    add_compile_options(/O2 /EHsc)
else()
    add_compile_options(-O3)
endif()

add_library(ptms_lib STATIC src/ptms.cpp)
target_include_directories(ptms_lib PUBLIC include PRIVATE src)
target_link_libraries(ptms_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(benchmark benchmark/benchmark.cpp)
target_include_directories(benchmark PRIVATE src)
target_link_libraries(benchmark PRIVATE ptms_lib)

enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE ptms_lib)
add_test(NAME correctness COMMAND test_correctness)

if(NOT MSVC)
    target_compile_options(ptms_lib         PRIVATE -Wall -Wextra)
    target_compile_options(benchmark        PRIVATE -Wall -Wextra)
    target_compile_options(test_correctness PRIVATE -Wall -Wextra)
endif()
