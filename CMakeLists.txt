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

# =====================================================================
#                       OPENMP
# =====================================================================
if(MINGW)
    message(STATUS "OpenMP: MinGW detected, using direct path")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.dll.a
        OUTPUT_VARIABLE GOMP_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GOMP_PATH OR GOMP_PATH STREQUAL "libgomp.dll.a")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a
            OUTPUT_VARIABLE GOMP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    message(STATUS "OpenMP: libgomp -> ${GOMP_PATH}")
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-fopenmp"
        INTERFACE_LINK_LIBRARIES "${GOMP_PATH}"
    )
else()
    find_package(OpenMP REQUIRED)
endif()

# =====================================================================
#                       SIMD SELECTION
# =====================================================================
set(SIMD "AUTO" CACHE STRING "SIMD level: AUTO, AVX512, AVX2, SCALAR")
set_property(CACHE SIMD PROPERTY STRINGS AUTO AVX512 AVX2 SCALAR)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

if(SIMD STREQUAL "AUTO")
    message(STATUS "SIMD: auto-detecting...")
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
        message(STATUS "SIMD: -march=native (CPU probed at compile time)")
    else()
        message(STATUS "SIMD: AVX2 (fallback, no -march=native)")
        if(MSVC)
            add_compile_options(/arch:AVX2)
        else()
            add_compile_options(-mavx2)
        endif()
    endif()
elseif(SIMD STREQUAL "AVX512")
    message(STATUS "SIMD: AVX-512 (user override)")
    if(MSVC)
        add_compile_options(/arch:AVX512)
    else()
        check_cxx_compiler_flag("-mavx512f" HAS_MAVX512)
        if(HAS_MAVX512)
            add_compile_options(-mavx512f)
        else()
            message(STATUS "  -> not supported, falling back to SCALAR")
        endif()
    endif()
elseif(SIMD STREQUAL "AVX2")
    message(STATUS "SIMD: AVX2 (user override)")
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        check_cxx_compiler_flag("-mavx2" HAS_MAVX2)
        if(HAS_MAVX2)
            add_compile_options(-mavx2)
        else()
            message(STATUS "  -> not supported, falling back to SCALAR")
        endif()
    endif()
elseif(SIMD STREQUAL "SCALAR")
    message(STATUS "SIMD: scalar (user override)")
else()
    message(FATAL_ERROR "Unknown SIMD mode: ${SIMD}")
endif()

# =====================================================================
#                       OPTIMIZATION
# =====================================================================
if(MSVC)
    add_compile_options(/O2 /EHsc)
else()
    add_compile_options(-O3)
endif()

# =====================================================================
#                       LIBRARY
# =====================================================================
add_library(ptms_lib STATIC src/ptms.cpp)
target_include_directories(ptms_lib PUBLIC include PRIVATE src)
target_link_libraries(ptms_lib PUBLIC OpenMP::OpenMP_CXX)

# =====================================================================
#                       BENCHMARK
# =====================================================================
add_executable(benchmark benchmark/benchmark.cpp)
target_include_directories(benchmark PRIVATE src)
target_link_libraries(benchmark PRIVATE ptms_lib)

# =====================================================================
#                       TBB (Intel NuGet package)
# =====================================================================
# Auto-detect the latest version of inteltbb.devel.win / inteltbb.redist.win
# in the user's NuGet cache. If not found, tbb::parallel_sort is skipped.

file(GLOB _TBB_DEV_ROOTS "$ENV{USERPROFILE}/.nuget/packages/inteltbb.devel.win/*")
file(GLOB _TBB_REDIST_ROOTS "$ENV{USERPROFILE}/.nuget/packages/inteltbb.redist.win/*")
list(SORT _TBB_DEV_ROOTS)
list(SORT _TBB_REDIST_ROOTS)

if(_TBB_DEV_ROOTS)
    list(GET _TBB_DEV_ROOTS -1 _TBB_DEV_ROOT)   # last = newest
    set(_TBB_INCLUDE "${_TBB_DEV_ROOT}/build/native/include")
    set(_TBB_LIB     "${_TBB_DEV_ROOT}/build/native/win-x64/tbb12.lib")
endif()

if(_TBB_REDIST_ROOTS)
    list(GET _TBB_REDIST_ROOTS -1 _TBB_REDIST_ROOT)
    file(GLOB_RECURSE _TBB_DLLS "${_TBB_REDIST_ROOT}/*/tbb12.dll")
    if(_TBB_DLLS)
        list(GET _TBB_DLLS 0 _TBB_DLL)
    endif()
endif()

if(_TBB_INCLUDE AND _TBB_LIB AND EXISTS "${_TBB_INCLUDE}/tbb/parallel_sort.h")
    add_library(TBB::tbb UNKNOWN IMPORTED)
    set_target_properties(TBB::tbb PROPERTIES
        IMPORTED_LOCATION "${_TBB_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_TBB_INCLUDE}"
    )
    target_link_libraries(benchmark PRIVATE TBB::tbb)
    target_compile_definitions(benchmark PRIVATE HAVE_TBB=1)
    message(STATUS "TBB: enabled (Intel NuGet)")
    message(STATUS "  include: ${_TBB_INCLUDE}")
    message(STATUS "  library: ${_TBB_LIB}")

    # Copy the TBB runtime DLL next to benchmark.exe so it runs without
    # having to set PATH manually.
    if(_TBB_DLL AND EXISTS "${_TBB_DLL}")
        add_custom_command(TARGET benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_TBB_DLL}" "$<TARGET_FILE_DIR:benchmark>"
            COMMENT "Copying tbb12.dll next to benchmark.exe"
        )
        message(STATUS "  dll:     ${_TBB_DLL}")
    else()
        message(STATUS "  dll:     not found in inteltbb.redist.win (install it if the .exe fails to start)")
    endif()
else()
    message(STATUS "TBB: NOT enabled — headers or lib not found")
    message(STATUS "  look in: $ENV{USERPROFILE}/.nuget/packages/inteltbb.devel.win/")
    message(STATUS "  install via VS NuGet: 'inteltbb.devel.win' + 'inteltbb.redist.win'")
endif()

# =====================================================================
#                       TESTS
# =====================================================================
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE ptms_lib)
add_test(NAME correctness COMMAND test_correctness)

# =====================================================================
#                       WARNINGS
# =====================================================================
if(NOT MSVC)
    target_compile_options(ptms_lib         PRIVATE -Wall -Wextra)
    target_compile_options(benchmark        PRIVATE -Wall -Wextra)
    target_compile_options(test_correctness PRIVATE -Wall -Wextra)
endif()
