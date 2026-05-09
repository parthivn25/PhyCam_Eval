/**
 * test_hdr.cpp — Catch2 unit tests for HDRCompressionOperator.
 */
#include "catch2/catch_amalgamated.hpp"
#include "phycam/hdr.hpp"
#include <cmath>

using namespace phycam;

static ImageBuffer make_random(int C, int H, int W, unsigned int seed = 0) {
    ImageBuffer buf(C, H, W);
    unsigned int s = seed;
    for (float& v : buf.data) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(s) / 4294967295.0f;
    }
    return buf;
}

TEST_CASE("HDRCompressionOperator: output shape unchanged", "[hdr]") {
    auto img = make_random(3, 64, 64);
    HDRCompressionOperator op(0.5);
    auto out = op.apply_copy(img);
    REQUIRE(out.channels == img.channels);
    REQUIRE(out.height   == img.height);
    REQUIRE(out.width    == img.width);
}

TEST_CASE("HDRCompressionOperator: output in [0,1]", "[hdr]") {
    auto img = make_random(3, 64, 64, 7);
    HDRCompressionOperator op(0.3);
    auto out = op.apply_copy(img);
    for (float v : out.data) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

TEST_CASE("HDRCompressionOperator: invalid beta throws", "[hdr]") {
    REQUIRE_THROWS_AS(HDRCompressionOperator(0.0),  std::invalid_argument);
    REQUIRE_THROWS_AS(HDRCompressionOperator(-0.5), std::invalid_argument);
}

TEST_CASE("HDRCompressionOperator: compression_ratio_db negative for beta<1", "[hdr]") {
    HDRCompressionOperator op(0.5);
    REQUIRE(op.compression_ratio_db() < 0.0);
}

TEST_CASE("HDRCompressionOperator: beta=1 is identity", "[hdr]") {
    auto img = make_random(3, 64, 64, 13);
    HDRCompressionOperator op(1.0);
    auto out = op.apply_copy(img);
    for (std::size_t i = 0; i < img.data.size(); ++i) {
        REQUIRE(std::abs(img.data[i] - out.data[i]) < 1e-4f);
    }
}
