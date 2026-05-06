/**
 * benchmark.cpp — C++ vs Python timing for degradation operators.
 *
 * Run: ./phycam_benchmark [image_size] [n_repeats]
 * Defaults: 640x640, 50 repeats.
 *
 * Outputs timing in microseconds per image, suitable for comparison
 * against the Python numpy/torch implementations.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <functional>

#include "phycam/types.hpp"
#include "phycam/optical.hpp"
#include "phycam/hdr.hpp"
#include "phycam/noise.hpp"

using namespace phycam;
using Clock = std::chrono::high_resolution_clock;

ImageBuffer make_random_image(int C, int H, int W) {
    ImageBuffer buf(C, H, W);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& v : buf.data) v = dist(rng);
    return buf;
}

struct BenchResult {
    std::string name;
    double mean_us;
    double std_us;
    double throughput_mpx_s;  // megapixels per second
};

BenchResult bench(const std::string& name,
                  const ImageBuffer& img,
                  std::function<void()> fn,
                  int n_repeats = 50) {
    // Warmup
    for (int i = 0; i < 3; ++i) fn();

    std::vector<double> times;
    times.reserve(n_repeats);
    for (int i = 0; i < n_repeats; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double mean = 0.0;
    for (double t : times) mean += t;
    mean /= times.size();

    double var = 0.0;
    for (double t : times) { double d = t - mean; var += d * d; }
    var /= times.size();

    long pixels = (long)img.channels * img.height * img.width;
    double mpx_s = (pixels / 1e6) / (mean / 1e6);  // Mpx/s

    return {name, mean, std::sqrt(var), mpx_s};
}

int main(int argc, char* argv[]) {
    int SIZE     = (argc > 1) ? std::stoi(argv[1]) : 640;
    int REPEATS  = (argc > 2) ? std::stoi(argv[2]) : 50;

    std::cout << "\n=== phycam-eval C++ degradation benchmark ===\n"
              << "Image size : " << SIZE << "x" << SIZE << " (3 channels)\n"
              << "Repeats    : " << REPEATS << "\n\n"
              << std::left << std::setw(30) << "Operator"
              << std::right << std::setw(12) << "Mean (µs)"
              << std::setw(12) << "Std (µs)"
              << std::setw(16) << "Throughput"
              << "\n" << std::string(72, '-') << "\n";

    auto img_orig = make_random_image(3, SIZE, SIZE);

    // Helper: each benchmark makes a fresh copy to measure apply() not copy()
    auto run = [&](const std::string& name, auto fn_make_op) {
        auto op = fn_make_op();
        auto r = bench(name, img_orig, [&] {
            auto img = img_orig;   // cheap copy
            op.apply(img);
        }, REPEATS);
        std::cout << std::left  << std::setw(30) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.mean_us
                  << std::setw(12) << r.std_us
                  << std::setw(14) << std::setprecision(2) << r.throughput_mpx_s << " Mpx/s\n";
    };

    run("DefocusOperator(alpha=1.5)",      [] { return DefocusOperator(1.5); });
    run("DefocusOperator(alpha=3.0)",      [] { return DefocusOperator(3.0); });
    run("AstigmatismOperator(alpha=1.5)",  [] { return AstigmatismOperator(1.5, 0.0); });
    run("LowLightOperator(level=0.2)",     [] { return LowLightOperator(0.2, 2, 42); });
    run("HDRCompressionOperator(β=0.5)",   [] { return HDRCompressionOperator(0.5); });
    run("SensorNoiseOperator(ISO 3200)",   [] { return SensorNoiseOperator::from_iso(3200); });
    // Full pipeline: I_d = N( Q( A( I ) ) )
    {
        DefocusOperator     A(1.5);
        HDRCompressionOperator Q(0.5);
        SensorNoiseOperator N = SensorNoiseOperator::from_iso(800);
        auto r = bench("Full pipeline (A→Q→N)", img_orig, [&] {
            auto img = img_orig;
            A.apply(img);
            Q.apply(img);
            N.apply(img);
        }, REPEATS);
        std::cout << std::left  << std::setw(30) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.mean_us
                  << std::setw(12) << r.std_us
                  << std::setw(14) << std::setprecision(2) << r.throughput_mpx_s << " Mpx/s\n";
    }

    std::cout << "\nNote: compare against Python baseline by running:\n"
              << "  python3 scripts/benchmark_python.py\n\n";
    return 0;
}
