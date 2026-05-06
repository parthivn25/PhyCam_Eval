/**
 * phycam/types.hpp — shared type aliases and image layout conventions.
 *
 * Image layout throughout the C++ library:
 *   float* data with shape (C, H, W), row-major (C-contiguous).
 *   Values in [0.0f, 1.0f].
 *   Channels: 0=R, 1=G, 2=B (matches PyTorch CHW convention).
 */
#pragma once

#include <cstddef>
#include <complex>
#include <vector>

namespace phycam {

using Real    = double;          // use double for FFT precision
using Complex = std::complex<Real>;

/** Flat CHW image buffer — owns its data. */
struct ImageBuffer {
    int channels;
    int height;
    int width;
    std::vector<float> data;     // length = channels * height * width

    ImageBuffer() = default;
    ImageBuffer(int C, int H, int W)
        : channels(C), height(H), width(W), data(C * H * W, 0.0f) {}

    float& at(int c, int h, int w)       { return data[c * height * width + h * width + w]; }
    float  at(int c, int h, int w) const { return data[c * height * width + h * width + w]; }

    std::size_t num_pixels() const { return static_cast<std::size_t>(height) * width; }
    std::size_t size()       const { return data.size(); }
};

} // namespace phycam
