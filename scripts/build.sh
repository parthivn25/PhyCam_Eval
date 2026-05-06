#!/usr/bin/env bash
# Build the phycam_eval C++ extension + benchmark.
# Run from the project root: ./scripts/build.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CPP_DIR="$PROJECT_ROOT/cpp"
BUILD_DIR="$CPP_DIR/build"

echo "=== phycam-eval C++ build ==="
echo "Project root : $PROJECT_ROOT"
echo "Build dir    : $BUILD_DIR"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: no Python interpreter found in PATH" >&2
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
    JOBS="$(getconf _NPROCESSORS_ONLN)"
elif command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
else
    JOBS=4
fi

cmake "$CPP_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARK=ON \
    -DPython_EXECUTABLE="$PYTHON_BIN"

make -j"$JOBS" phycam_cpp phycam_benchmark

echo ""
echo "=== Build complete ==="
echo "Extension   : $PROJECT_ROOT/phycam_eval/degradations/phycam_cpp*.so"
echo "Benchmark   : $BUILD_DIR/phycam_benchmark"
echo "Python      : $PYTHON_BIN"
echo ""
echo "To verify:"
echo "  $PYTHON_BIN -c \"from phycam_eval.degradations import DefocusOperator; print(DefocusOperator(1.5))\""
echo "  ./cpp/build/phycam_benchmark"
