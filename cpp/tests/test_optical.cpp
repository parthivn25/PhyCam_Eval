/**
 * test_optical.cpp — Catch2 unit tests for phycam optical operators.
 */
#define CATCH_CONFIG_MAIN
#include "catch2/catch_amalgamated.hpp"
#include "phycam/optical.hpp"
#include <cmath>
#include <numeric>

using namespace phycam;
using Catch::Approx;

// ---- helpers ---------------------------------------------------------------

static ImageBuffer make_random_image(int C, int H, int W, unsigned int seed = 42) {
    ImageBuffer buf(C, H, W);
    unsigned int s = seed;
    for (float& v : buf.data) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(s) / 4294967295.0f;
    }
    return buf;
}

static double psnr(const ImageBuffer& a, const ImageBuffer& b) {
    double mse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a.data[i] - b.data[i];
        mse += d * d;
    }
    mse /= a.size();
    if (mse < 1e-12) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(1.0 / mse);
}

// ---- DefocusOperator -------------------------------------------------------

TEST_CASE("DefocusOperator: output shape unchanged", "[optical][defocus]") {
    auto img = make_random_image(3, 64, 64);
    DefocusOperator op(1.5);
    auto out = op.apply_copy(img);
    REQUIRE(out.channels == img.channels);
    REQUIRE(out.height   == img.height);
    REQUIRE(out.width    == img.width);
}

TEST_CASE("DefocusOperator: output clamped to [0,1]", "[optical][defocus]") {
    auto img = make_random_image(3, 64, 64);
    DefocusOperator op(2.5);
    auto out = op.apply_copy(img);
    for (float v : out.data) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

TEST_CASE("DefocusOperator: alpha=0 is identity", "[optical][defocus]") {
    auto img = make_random_image(3, 64, 64);
    DefocusOperator op(0.0);
    auto out = op.apply_copy(img);
    double p = psnr(img, out);
    // Should be essentially lossless (limited by float round-trip through FFT)
    REQUIRE(p > 50.0);
}

TEST_CASE("DefocusOperator: increasing alpha increases degradation", "[optical][defocus]") {
    auto img = make_random_image(3, 64, 64, 7);
    double prev_psnr = std::numeric_limits<double>::infinity();
    for (double alpha : {0.5, 1.0, 1.5, 2.0, 2.5}) {
        DefocusOperator op(alpha);
        auto out = op.apply_copy(img);
        double p = psnr(img, out);
        REQUIRE(p < prev_psnr);
        prev_psnr = p;
    }
}

TEST_CASE("DefocusOperator: negative alpha throws", "[optical][defocus]") {
    REQUIRE_THROWS_AS(DefocusOperator(-0.1), std::invalid_argument);
}

TEST_CASE("DefocusOperator: phase mask is quadratic in rho", "[optical][defocus]") {
    DefocusOperator op(2.0);
    auto phi = op.phase_mask(32, 32);
    // DC element (0,0): rho=0 → phi=0
    REQUIRE(phi[0] == Approx(0.0).margin(1e-10));
    // All values non-negative
    for (double v : phi) REQUIRE(v >= 0.0);
}

TEST_CASE("DefocusOperator: OTF magnitude is all-ones (phase only)", "[optical][defocus]") {
    DefocusOperator op(1.5);
    auto otf = op.otf_magnitude(32, 32);
    for (double v : otf) REQUIRE(v == Approx(1.0).margin(1e-10));
}

// ---- AstigmatismOperator ---------------------------------------------------

TEST_CASE("AstigmatismOperator: output shape and range", "[optical][astigmatism]") {
    auto img = make_random_image(3, 64, 64, 3);
    AstigmatismOperator op(1.5, 0.0);
    auto out = op.apply_copy(img);
    REQUIRE(out.channels == img.channels);
    for (float v : out.data) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

TEST_CASE("AstigmatismOperator: different theta_axis gives different result", "[optical][astigmatism]") {
    auto img = make_random_image(3, 64, 64, 5);
    AstigmatismOperator op0(2.0, 0.0);
    AstigmatismOperator op90(2.0, M_PI / 2.0);
    auto out0  = op0.apply_copy(img);
    auto out90 = op90.apply_copy(img);
    // Results should differ
    double max_diff = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i)
        max_diff = std::max(max_diff, std::abs((double)(out0.data[i] - out90.data[i])));
    REQUIRE(max_diff > 0.001);
}

// ---- LowLightOperator ------------------------------------------------------

TEST_CASE("LowLightOperator: output shape and range", "[optical][lowlight]") {
    auto img = make_random_image(3, 64, 64, 9);
    LowLightOperator op(0.3, 2, 42);
    auto out = op.apply_copy(img);
    REQUIRE(out.channels == img.channels);
    for (float v : out.data) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

TEST_CASE("LowLightOperator: invalid light_level throws", "[optical][lowlight]") {
    REQUIRE_THROWS_AS(LowLightOperator(0.0),  std::invalid_argument);
    REQUIRE_THROWS_AS(LowLightOperator(1.5),  std::invalid_argument);
}

TEST_CASE("LowLightOperator: bright preserves edge sharpness better than dark", "[optical][lowlight]") {
    // Use a high-contrast step edge (half black, half white).
    // Dark LowLightOperator applies stronger Butterworth blur AND more shot noise.
    // We measure PEAK gradient at the transition (noise-robust: noise adds uniform
    // variance whereas blurring reduces the peak).
    ImageBuffer img(1, 64, 64);
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c)
            img.at(0, r, c) = (c >= 32) ? 1.0f : 0.0f;

    LowLightOperator bright(0.9, 2, 1);
    LowLightOperator dark(0.05, 2, 1);   // very dark → very strong blur
    auto out_bright = bright.apply_copy(img);
    auto out_dark   = dark.apply_copy(img);

    // Row-average at each column to suppress shot noise, then find peak gradient.
    auto row_mean = [](const ImageBuffer& buf, int W) {
        std::vector<double> mean(W, 0.0);
        int H = buf.height;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                mean[c] += buf.at(0, r, c);
        for (auto& v : mean) v /= H;
        return mean;
    };

    auto mean_bright = row_mean(out_bright, 64);
    auto mean_dark   = row_mean(out_dark,   64);

    // Peak gradient = max |mean[c] - mean[c-1]| near edge (columns 24..40)
    double peak_bright = 0.0, peak_dark = 0.0;
    for (int c = 25; c < 40; ++c) {
        peak_bright = std::max(peak_bright, std::abs(mean_bright[c] - mean_bright[c - 1]));
        peak_dark   = std::max(peak_dark,   std::abs(mean_dark[c]   - mean_dark[c - 1]));
    }
    REQUIRE(peak_bright > peak_dark);
}
