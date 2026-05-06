/**
 * phycam/fft2d.hpp — 2-D FFT/IFFT wrapper around libfftw3.
 *
 * Provides a thin RAII wrapper: FFT2D manages one FFTW plan and its
 * associated input/output buffers.  Reusing the same plan for multiple
 * images of the same size avoids the per-call planning overhead.
 *
 * Usage:
 *   FFT2D fft(H, W);
 *   fft.forward(real_in);           // fills fft.freq[]
 *   fft.apply_transfer(H_complex);  // multiply in frequency domain
 *   auto result = fft.inverse();    // returns spatial-domain real array
 */
#pragma once

#include <complex>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>

#include "thirdparty/fftw3.h"
#include "phycam/types.hpp"

namespace phycam {

class FFT2D {
public:
    int H, W;
    int N;               // H * W

    // FFTW buffers (allocated with fftw_malloc for alignment)
    fftw_complex* in_buf  = nullptr;
    fftw_complex* out_buf = nullptr;

    fftw_plan plan_fwd = nullptr;
    fftw_plan plan_inv = nullptr;

    explicit FFT2D(int height, int width)
        : H(height), W(width), N(height * width)
    {
        in_buf  = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
        out_buf = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
        if (!in_buf || !out_buf) {
            fftw_free(in_buf);
            fftw_free(out_buf);
            in_buf = out_buf = nullptr;
            throw std::runtime_error("fftw_malloc failed");
        }

        plan_fwd = fftw_plan_dft_2d(H, W, in_buf, out_buf, FFTW_FORWARD,  FFTW_ESTIMATE);
        plan_inv = fftw_plan_dft_2d(H, W, in_buf, out_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!plan_fwd || !plan_inv) {
            if (plan_fwd) fftw_destroy_plan(plan_fwd);
            if (plan_inv) fftw_destroy_plan(plan_inv);
            fftw_free(in_buf);
            fftw_free(out_buf);
            plan_fwd = plan_inv = nullptr;
            in_buf = out_buf = nullptr;
            throw std::runtime_error("fftw_plan_dft_2d failed");
        }
    }

    ~FFT2D() {
        if (plan_fwd) fftw_destroy_plan(plan_fwd);
        if (plan_inv) fftw_destroy_plan(plan_inv);
        if (in_buf)   fftw_free(in_buf);
        if (out_buf)  fftw_free(out_buf);
    }

    // Non-copyable (owns FFTW plans)
    FFT2D(const FFT2D&) = delete;
    FFT2D& operator=(const FFT2D&) = delete;

    /** Load a real-valued (H,W) channel into in_buf (imaginary = 0). */
    void load_real(const float* channel) {
        for (int i = 0; i < N; ++i) {
            in_buf[i][0] = static_cast<Real>(channel[i]);
            in_buf[i][1] = 0.0;
        }
    }

    /** Forward FFT: in_buf → out_buf. */
    void forward() {
        fftw_execute(plan_fwd);
    }

    /**
     * Multiply out_buf by a complex transfer function H[i] in place,
     * then write result back to in_buf for the inverse step.
     *
     * H_real[i], H_imag[i]  — real and imaginary parts of H(ωᵢ)
     */
    void apply_transfer(const std::vector<Real>& H_real,
                        const std::vector<Real>& H_imag) {
        for (int i = 0; i < N; ++i) {
            Real ar = out_buf[i][0], ai = out_buf[i][1];
            Real hr = H_real[i],    hi = H_imag[i];
            // complex multiply: (ar + j ai)(hr + j hi)
            in_buf[i][0] = ar * hr - ai * hi;
            in_buf[i][1] = ar * hi + ai * hr;
        }
    }

    /**
     * Multiply out_buf by a PHASE-ONLY transfer function exp(j phi[i]).
     * More efficient than the general complex multiply when |H| = 1.
     */
    void apply_phase(const std::vector<Real>& phi) {
        for (int i = 0; i < N; ++i) {
            Real ar = out_buf[i][0], ai = out_buf[i][1];
            Real hr = std::cos(phi[i]), hi = std::sin(phi[i]);
            in_buf[i][0] = ar * hr - ai * hi;
            in_buf[i][1] = ar * hi + ai * hr;
        }
    }

    /**
     * Multiply out_buf by a REAL amplitude transfer function A[i].
     * Used for low-light Butterworth and HDR magnitude scaling.
     */
    void apply_amplitude(const std::vector<Real>& A) {
        for (int i = 0; i < N; ++i) {
            in_buf[i][0] = out_buf[i][0] * A[i];
            in_buf[i][1] = out_buf[i][1] * A[i];
        }
    }

    /**
     * Inverse FFT (in_buf → out_buf), normalise by N,
     * write real part to dst[], clamped to [lo, hi].
     */
    void inverse_to(float* dst, float lo = 0.0f, float hi = 1.0f) {
        fftw_execute(plan_inv);
        Real inv_N = 1.0 / N;
        for (int i = 0; i < N; ++i) {
            float v = static_cast<float>(out_buf[i][0] * inv_N);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            dst[i] = v;
        }
    }

    /**
     * Convenience: forward, apply phase, inverse — all in one call.
     * Writes result to dst (same size as channel: H*W floats).
     */
    void transform_phase(const float* channel, const std::vector<Real>& phi, float* dst) {
        load_real(channel);
        forward();
        apply_phase(phi);
        inverse_to(dst);
    }

    /**
     * Convenience: forward, apply amplitude, inverse.
     */
    void transform_amplitude(const float* channel,
                              const std::vector<Real>& amp,
                              float* dst) {
        load_real(channel);
        forward();
        apply_amplitude(amp);
        inverse_to(dst);
    }
};

// ---- Frequency grid helpers -----------------------------------------------

/**
 * Fill fy[i], fx[i] with normalised frequency coordinates for pixel i = r*W+c.
 * Both in [-0.5, 0.5).  DC is at index 0 (FFTW convention: no fftshift needed).
 */
inline void freq_grid(int H, int W,
                      std::vector<Real>& fy,
                      std::vector<Real>& fx) {
    fy.resize(H * W);
    fx.resize(H * W);
    for (int r = 0; r < H; ++r) {
        Real fyr = (r < (H + 1) / 2) ? (Real)r / H : (Real)(r - H) / H;
        for (int c = 0; c < W; ++c) {
            Real fxc = (c < (W + 1) / 2) ? (Real)c / W : (Real)(c - W) / W;
            fy[r * W + c] = fyr;
            fx[r * W + c] = fxc;
        }
    }
}

/** Radial frequency rho[i] = sqrt(fx[i]^2 + fy[i]^2). */
inline std::vector<Real> radial_freq(const std::vector<Real>& fy,
                                     const std::vector<Real>& fx) {
    std::size_t N = fy.size();
    std::vector<Real> rho(N);
    for (std::size_t i = 0; i < N; ++i)
        rho[i] = std::sqrt(fy[i] * fy[i] + fx[i] * fx[i]);
    return rho;
}

} // namespace phycam
