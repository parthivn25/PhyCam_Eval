/**
 * test_noise.cpp — Catch2 unit tests for SensorNoiseOperator.
 */
#include "catch2/catch_amalgamated.hpp"
#include "phycam/noise.hpp"
#include <cmath>
#include <numeric>

using namespace phycam;
using Catch::Approx;

static ImageBuffer make_flat(int C, int H, int W, float value) {
    ImageBuffer buf(C, H, W);
    std::fill(buf.data.begin(), buf.data.end(), value);
    return buf;
}

TEST_CASE("SensorNoiseOperator: output shape unchanged", "[noise]") {
    auto img = make_flat(3, 64, 64, 0.5f);
    SensorNoiseOperator op(1.0);
    auto out = op.apply_copy(img);
    REQUIRE(out.channels == img.channels);
    REQUIRE(out.height   == img.height);
    REQUIRE(out.width    == img.width);
}

TEST_CASE("SensorNoiseOperator: output in [0,1] when clip=true", "[noise]") {
    auto img = make_flat(3, 64, 64, 0.5f);
    SensorNoiseOperator op(32.0, 0.002, true);
    auto out = op.apply_copy(img);
    for (float v : out.data) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

TEST_CASE("SensorNoiseOperator: invalid gain throws", "[noise]") {
    REQUIRE_THROWS_AS(SensorNoiseOperator(0.0),  std::invalid_argument);
    REQUIRE_THROWS_AS(SensorNoiseOperator(-1.0), std::invalid_argument);
}

TEST_CASE("SensorNoiseOperator: invalid read_noise_std throws", "[noise]") {
    REQUIRE_THROWS_AS(SensorNoiseOperator(1.0, -0.01), std::invalid_argument);
}

TEST_CASE("SensorNoiseOperator: noise variance increases with gain", "[noise]") {
    std::vector<double> variances;
    for (double gain : {1.0, 4.0, 16.0, 64.0}) {
        auto img = make_flat(3, 128, 128, 0.5f);
        SensorNoiseOperator op(gain, 0.001, false, 99);
        auto out = op.apply_copy(img);
        // Variance = E[(X - E[X])^2]
        double mean = 0.0;
        for (float v : out.data) mean += v;
        mean /= out.size();
        double var = 0.0;
        for (float v : out.data) { double d = v - mean; var += d * d; }
        var /= out.size();
        variances.push_back(var);
    }
    for (int i = 0; i + 1 < (int)variances.size(); ++i)
        REQUIRE(variances[i] < variances[i + 1]);
}

TEST_CASE("SensorNoiseOperator: from_iso sets correct gain", "[noise]") {
    auto op = SensorNoiseOperator::from_iso(3200, 100);
    REQUIRE(op.gain == Approx(32.0 * 5e-5).epsilon(1e-9));
}

TEST_CASE("SensorNoiseOperator: snr_db decreases with gain", "[noise]") {
    double prev = std::numeric_limits<double>::infinity();
    for (double gain : {1.0, 4.0, 16.0, 64.0}) {
        SensorNoiseOperator op(gain);
        double snr = op.snr_db(0.5);
        REQUIRE(snr < prev);
        prev = snr;
    }
}

TEST_CASE("SensorNoiseOperator: zero gain near-identity", "[noise]") {
    // Extremely low gain + low read noise → barely any noise
    auto img = make_flat(1, 128, 128, 0.5f);
    SensorNoiseOperator op(1e-6, 1e-6, false, 7);
    auto out = op.apply_copy(img);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i)
        max_diff = std::max(max_diff, std::abs((double)(img.data[i] - out.data[i])));
    REQUIRE(max_diff < 0.01);
}

TEST_CASE("SensorNoiseOperator: shot and read noise variances add in quadrature", "[noise]") {
    auto img = make_flat(1, 256, 256, 0.5f);
    SensorNoiseOperator op(0.04, 0.05, false, 7);
    auto out = op.apply_copy(img);

    double mean = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i)
        mean += (out.data[i] - img.data[i]);
    mean /= img.size();

    double var = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i) {
        double d = (out.data[i] - img.data[i]) - mean;
        var += d * d;
    }
    var /= img.size();

    double expected_var = 0.04 * 0.5 + 0.05 * 0.05;
    REQUIRE(var == Approx(expected_var).margin(0.003));
}
