/**
 * phycam/hdr.hpp — HDR dynamic range compression operator.
 *
 * From the ODRC formalism (Nair et al., 2026), ported to camera images:
 *   Q_β(I) = F⁻¹ { sign(F(I)) · |F(I)|^β }
 *
 * β=1 → identity.  β<1 → compression.  β>1 → expansion.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "phycam/types.hpp"
#include "phycam/fft2d.hpp"

namespace phycam {

class HDRCompressionOperator {
public:
    double beta;
    bool   per_channel;
    double eps;

    explicit HDRCompressionOperator(double beta = 0.5,
                                    bool per_channel = true,
                                    double eps = 1e-10)
        : beta(beta), per_channel(per_channel), eps(eps)
    {
        if (beta <= 0.0)
            throw std::invalid_argument("HDRCompressionOperator: beta must be > 0");
    }

    /** Apply Q_β to a single (H,W) channel.  Result is normalised to [0,1]. */
    void apply_channel(const float* src, float* dst, int H, int W) const {
        int N = H * W;
        FFT2D fft(H, W);
        fft.load_real(src);
        fft.forward();

        // Multiply out_buf by sign(F(I)) * |F(I)|^(beta-1)
        // i.e. new_F[i] = F[i] / |F[i]| * |F[i]|^beta
        //               = F[i] * |F[i]|^(beta-1)
        for (int i = 0; i < N; ++i) {
            Real ar = fft.out_buf[i][0];
            Real ai = fft.out_buf[i][1];
            Real mag = std::sqrt(ar * ar + ai * ai);
            if (mag < eps) {
                fft.in_buf[i][0] = 0.0;
                fft.in_buf[i][1] = 0.0;
            } else {
                Real scale = std::pow(mag, beta - 1.0);
                fft.in_buf[i][0] = ar * scale;
                fft.in_buf[i][1] = ai * scale;
            }
        }

        // Inverse FFT (in_buf already set above, bypass apply_transfer)
        // Need direct execute on plan_inv with in_buf → out_buf
        fftw_execute(fft.plan_inv);

        // Normalise and extract real part
        Real inv_N = 1.0 / N;
        std::vector<float> tmp(N);
        float tmin = 1e38f, tmax = -1e38f;
        for (int i = 0; i < N; ++i) {
            tmp[i] = static_cast<float>(fft.out_buf[i][0] * inv_N);
            if (tmp[i] < tmin) tmin = tmp[i];
            if (tmp[i] > tmax) tmax = tmp[i];
        }
        // Normalise output to [0,1]
        float range = tmax - tmin;
        if (range < 1e-8f) range = 1.0f;
        for (int i = 0; i < N; ++i)
            dst[i] = std::max(0.0f, std::min(1.0f, (tmp[i] - tmin) / range));
    }

    ImageBuffer& apply(ImageBuffer& buf) const {
        int H = buf.height, W = buf.width;
        int stride = H * W;

        bool use_per_channel = per_channel || (buf.channels < 3);
        if (use_per_channel) {
            for (int c = 0; c < buf.channels; ++c) {
                float* ch = buf.data.data() + c * stride;
                std::vector<float> tmp(stride);
                apply_channel(ch, tmp.data(), H, W);
                std::copy(tmp.begin(), tmp.end(), ch);
            }
        } else {
            // Luminance-only: compute Y, compress, rescale channels
            std::vector<float> Y(stride, 0.0f);
            const float kr = 0.299f, kg = 0.587f, kb = 0.114f;
            float* R = buf.data.data();
            float* G = buf.data.data() + stride;
            float* B = buf.data.data() + 2 * stride;
            for (int i = 0; i < stride; ++i)
                Y[i] = kr * R[i] + kg * G[i] + kb * B[i];

            std::vector<float> Y_comp(stride);
            apply_channel(Y.data(), Y_comp.data(), H, W);

            // Scale each channel by Y_comp / Y
            for (int i = 0; i < stride; ++i) {
                float ratio = (Y[i] > 1e-6f) ? std::min(Y_comp[i] / Y[i], 2.0f) : 1.0f;
                R[i] = std::max(0.0f, std::min(1.0f, R[i] * ratio));
                G[i] = std::max(0.0f, std::min(1.0f, G[i] * ratio));
                B[i] = std::max(0.0f, std::min(1.0f, B[i] * ratio));
            }
        }
        return buf;
    }

    ImageBuffer apply_copy(const ImageBuffer& buf) const {
        ImageBuffer out = buf;
        apply(out);
        return out;
    }

    double compression_ratio_db() const {
        return 20.0 * std::log10(beta);
    }
};

} // namespace phycam
