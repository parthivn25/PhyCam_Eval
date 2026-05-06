/**
 * phycam/optical.hpp — physics-grounded optical degradation operators.
 *
 * Three operators, all instances of the spectral-phase template:
 *   A_φ(I) = F⁻¹ { F(I) · exp(j φ(ω)) }
 *
 * Each derives φ from Fourier optics (Goodman, Intro to Fourier Optics, Ch.6).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "phycam/types.hpp"
#include "phycam/fft2d.hpp"

namespace phycam {

// ---------------------------------------------------------------------------
// DefocusOperator
// ---------------------------------------------------------------------------

/**
 * Defocus via quadratic pupil phase.
 *
 * Physical derivation:
 *   Wavefront error for defocus:  W(u,v) = W₂₀ · r²/R²
 *   Pupil function:  P(u,v) = circ(r/R) · exp(j 2π W₂₀ r²/R²)
 *   PST equivalent:  φ(ρ) = α · ρ²
 *
 * @param alpha  Phase strength (≥0). 0 = identity. ~π = visible blur.
 * @param normalize_freq  If true, ρ normalised to [0,1] before applying φ.
 */
class DefocusOperator {
public:
    double alpha;
    bool   normalize_freq;

    explicit DefocusOperator(double alpha = 1.0, bool normalize_freq = true)
        : alpha(alpha), normalize_freq(normalize_freq)
    {
        if (alpha < 0.0)
            throw std::invalid_argument("DefocusOperator: alpha must be >= 0");
    }

    /**
     * Build the phase mask φ(ρ) = alpha * ρ² for an (H,W) image.
     * Returned vector has length H*W.
     */
    std::vector<Real> phase_mask(int H, int W) const {
        std::vector<Real> fy, fx;
        freq_grid(H, W, fy, fx);
        auto rho = radial_freq(fy, fx);

        if (normalize_freq) {
            Real rho_max = 0.0;
            for (auto r : rho) if (r > rho_max) rho_max = r;
            if (rho_max < 1e-10) rho_max = 1.0;
            for (auto& r : rho) r /= rho_max;
        }

        std::vector<Real> phi(H * W);
        for (int i = 0; i < H * W; ++i)
            phi[i] = alpha * rho[i] * rho[i];
        return phi;
    }

    /** Apply defocus to a (C,H,W) image buffer in-place.  Returns ref to buf. */
    ImageBuffer& apply(ImageBuffer& buf) const {
        int H = buf.height, W = buf.width;
        auto phi = phase_mask(H, W);
        FFT2D fft(H, W);

        std::vector<float> tmp(H * W);
        for (int c = 0; c < buf.channels; ++c) {
            float* channel = buf.data.data() + c * H * W;
            // transform_phase writes directly back to channel
            fft.transform_phase(channel, phi, tmp.data());
            std::copy(tmp.begin(), tmp.end(), channel);
        }
        return buf;
    }

    /** Apply to a copy, leave original unchanged. */
    ImageBuffer apply_copy(const ImageBuffer& buf) const {
        ImageBuffer out = buf;
        apply(out);
        return out;
    }

    /**
     * OTF magnitude (MTF) for this defocus strength.
     * Returns a flat (H*W) vector of |exp(j phi)| = 1 everywhere
     * (phase-only filter).  For a physically complete OTF including
     * the pupil aperture, see otf_with_aperture().
     */
    std::vector<Real> otf_magnitude(int H, int W) const {
        // Phase-only: |H(ω)| = 1 ∀ ω
        return std::vector<Real>(H * W, 1.0);
    }
};


// ---------------------------------------------------------------------------
// AstigmatismOperator
// ---------------------------------------------------------------------------

/**
 * Astigmatism via PAGE-derived axis-dependent quadratic phase.
 *
 * Zernike Z₂⁺² (oblique astigmatism):  W(u,v) = W₂₂(u² - v²)
 * PAGE spectral phase:  φ(ρ,θ) = α·cos²(θ - θ_axis) · ρ²
 *
 * @param alpha      Maximum phase strength.
 * @param theta_axis Blur axis angle in radians (0 = horizontal).
 */
class AstigmatismOperator {
public:
    double alpha;
    double theta_axis;

    explicit AstigmatismOperator(double alpha = 1.0, double theta_axis = 0.0)
        : alpha(alpha), theta_axis(theta_axis) {}

    std::vector<Real> phase_mask(int H, int W) const {
        std::vector<Real> fy, fx;
        freq_grid(H, W, fy, fx);
        auto rho = radial_freq(fy, fx);

        Real rho_max = 0.0;
        for (auto r : rho) if (r > rho_max) rho_max = r;
        if (rho_max < 1e-10) rho_max = 1.0;

        std::vector<Real> phi(H * W);
        for (int i = 0; i < H * W; ++i) {
            Real rho_n = rho[i] / rho_max;
            Real theta = std::atan2(fy[i], fx[i]);
            Real cos_t = std::cos(theta - theta_axis);
            phi[i] = alpha * cos_t * cos_t * rho_n * rho_n;
        }
        return phi;
    }

    ImageBuffer& apply(ImageBuffer& buf) const {
        int H = buf.height, W = buf.width;
        auto phi = phase_mask(H, W);
        FFT2D fft(H, W);
        std::vector<float> tmp(H * W);
        for (int c = 0; c < buf.channels; ++c) {
            float* channel = buf.data.data() + c * H * W;
            fft.transform_phase(channel, phi, tmp.data());
            std::copy(tmp.begin(), tmp.end(), channel);
        }
        return buf;
    }

    ImageBuffer apply_copy(const ImageBuffer& buf) const {
        ImageBuffer out = buf;
        apply(out);
        return out;
    }
};


// ---------------------------------------------------------------------------
// LowLightOperator
// ---------------------------------------------------------------------------

/**
 * Low-light degradation: VEViD-inspired photon-starvation model.
 *
 * Butterworth low-pass (amplitude, not phase):
 *   H_ll(ρ) = 1 / (1 + (ρ / ρ_c)^{2n})
 * where ρ_c = sqrt(light_level)  (cutoff narrows as light drops).
 *
 * Shot noise is added after filtering:
 *   σ_shot = 0.05 / sqrt(light_level)
 *
 * @param light_level  In (0,1].  1.0 = well-lit, 0.05 = very dark.
 * @param order        Butterworth order (controls roll-off). Default 2.
 */
class LowLightOperator {
public:
    double light_level;
    int    order;
    unsigned int rng_seed;

    explicit LowLightOperator(double light_level = 0.5,
                               int order = 2,
                               unsigned int seed = 42)
        : light_level(light_level), order(order), rng_seed(seed)
    {
        if (light_level <= 0.0 || light_level > 1.0)
            throw std::invalid_argument("LowLightOperator: light_level must be in (0,1]");
    }

    std::vector<Real> amplitude_mask(int H, int W) const {
        std::vector<Real> fy, fx;
        freq_grid(H, W, fy, fx);
        auto rho = radial_freq(fy, fx);

        Real rho_max = 0.0;
        for (auto r : rho) if (r > rho_max) rho_max = r;
        if (rho_max < 1e-10) rho_max = 1.0;

        double rho_c = std::sqrt(light_level);
        int    two_n = 2 * order;
        std::vector<Real> amp(H * W);
        for (int i = 0; i < H * W; ++i) {
            double rho_n = rho[i] / rho_max;
            double ratio = rho_n / rho_c;
            double denom = 1.0 + std::pow(ratio, two_n);
            amp[i] = 1.0 / denom;
        }
        return amp;
    }

    ImageBuffer& apply(ImageBuffer& buf) const {
        static constexpr double kPi = 3.14159265358979323846;
        int H = buf.height, W = buf.width;
        auto amp = amplitude_mask(H, W);
        FFT2D fft(H, W);

        // Simple LCG for shot noise (reproducible, no <random> dependency issues)
        double noise_std = 0.05 / std::sqrt(light_level);
        auto lcg_next = [](unsigned int& s) -> double {
            s = s * 1664525u + 1013904223u;
            return s / 4294967296.0;  // [0, 1)
        };
        unsigned int seed = rng_seed;

        std::vector<float> tmp(H * W);
        for (int c = 0; c < buf.channels; ++c) {
            float* channel = buf.data.data() + c * H * W;
            fft.transform_amplitude(channel, amp, tmp.data());

            // Add shot noise (Box-Muller for approximate Gaussian)
            for (int i = 0; i < H * W; i += 2) {
                double u1 = std::max(lcg_next(seed), 1e-12);
                double u2 = lcg_next(seed);
                double mag = std::sqrt(-2.0 * std::log(u1)) * noise_std;
                double ang = 2.0 * kPi * u2;
                float n1 = static_cast<float>(mag * std::cos(ang));
                float n2 = static_cast<float>(mag * std::sin(ang));
                tmp[i]     = std::max(0.0f, std::min(1.0f, tmp[i]     + n1));
                if (i + 1 < H * W)
                    tmp[i + 1] = std::max(0.0f, std::min(1.0f, tmp[i + 1] + n2));
            }
            std::copy(tmp.begin(), tmp.end(), channel);
        }
        return buf;
    }

    ImageBuffer apply_copy(const ImageBuffer& buf) const {
        ImageBuffer out = buf;
        apply(out);
        return out;
    }
};

} // namespace phycam
