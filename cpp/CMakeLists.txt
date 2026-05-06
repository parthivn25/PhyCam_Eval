cmake_minimum_required(VERSION 3.18)
project(phycam_eval_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ---- Optimisation flags ----------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# ---- Include dirs ----------------------------------------------------------
set(PHYCAM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- FFTW3 -----------------------------------------------------------------
# The runtime .so is on the system; we provide our own minimal header stub.
find_library(FFTW3_LIB
    NAMES fftw3 libfftw3 fftw3.so.3 libfftw3.so.3
    PATHS
        /opt/homebrew/lib                          # macOS Homebrew (arm64)
        /usr/local/lib                             # macOS Homebrew (x86_64)
        /usr/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
        /usr/lib/arm-linux-gnueabihf
    NO_DEFAULT_PATH)
if(NOT FFTW3_LIB)
    # Fall back: use versioned .so/.dylib directly
    find_file(FFTW3_LIB
        NAMES libfftw3.so.3 libfftw3.so.3.5.8 libfftw3.dylib libfftw3.3.dylib
        PATHS
            /opt/homebrew/lib /usr/local/lib
            /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
endif()
if(NOT FFTW3_LIB)
    message(FATAL_ERROR "Could not find libfftw3. "
        "macOS: brew install fftw  |  Linux: sudo apt-get install libfftw3-dev")
endif()
message(STATUS "FFTW3 library: ${FFTW3_LIB}")

# FFTW3 include directory (real headers, not our stub)
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include
    NO_DEFAULT_PATH)
if(FFTW3_INCLUDE_DIR)
    message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")
endif()

# ---- pybind11 --------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try system pybind11 first, then pip-installed
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Use pip-installed pybind11
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Run: pip install pybind11")
    endif()
endif()

# ---- Python extension module -----------------------------------------------
pybind11_add_module(phycam_cpp
    bindings/phycam_bindings.cpp
)

target_include_directories(phycam_cpp PRIVATE ${PHYCAM_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR})
target_link_libraries(phycam_cpp PRIVATE ${FFTW3_LIB})

# Install the .so next to the Python package so `import phycam_cpp` works
set_target_properties(phycam_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../phycam_eval/degradations
)

# ---- C++ unit tests --------------------------------------------------------
option(BUILD_TESTS "Build C++ unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Catch2 v3 — amalgamated single-file in cpp/tests/catch2/
    set(CATCH2_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2/catch_amalgamated.hpp)

    if(EXISTS ${CATCH2_HEADER})
        add_executable(phycam_tests
            tests/test_optical.cpp
            tests/test_hdr.cpp
            tests/test_noise.cpp
            tests/catch2/catch_amalgamated.cpp
        )
        target_include_directories(phycam_tests PRIVATE
            ${PHYCAM_INCLUDE_DIR}
            ${FFTW3_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2
        )
        target_link_libraries(phycam_tests PRIVATE ${FFTW3_LIB})
        # Remove -ffast-math for tests: it makes std::numeric_limits<>::infinity() UB
        target_compile_options(phycam_tests PRIVATE -fno-fast-math)
        add_test(NAME PhycamTests COMMAND phycam_tests)
    else()
        message(WARNING
            "Catch2 amalgam not found at ${CATCH2_HEADER}. "
            "Download from https://github.com/catchorg/Catch2/releases "
            "and place in cpp/tests/catch2/.")
    endif()
endif()

# ---- Benchmark executable --------------------------------------------------
option(BUILD_BENCHMARK "Build C++ vs Python benchmark" ON)
if(BUILD_BENCHMARK)
    add_executable(phycam_benchmark benchmark/benchmark.cpp)
    target_include_directories(phycam_benchmark PRIVATE ${PHYCAM_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR})
    target_link_libraries(phycam_benchmark PRIVATE ${FFTW3_LIB})
endif()

message(STATUS "")
message(STATUS "phycam-eval C++ build configured:")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags  : ${CMAKE_CXX_FLAGS}")
message(STATUS "  FFTW3      : ${FFTW3_LIB}")
message(STATUS "  Extension  : phycam_cpp  →  phycam_eval/degradations/")
message(STATUS "")
