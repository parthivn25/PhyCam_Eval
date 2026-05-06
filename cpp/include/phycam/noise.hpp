/**
 * phycam/noise.hpp — mixed Poisson-Gaussian sensor noise model.
 *
 * N(I) = Poisson(I/g)·g + Normal(0, σ_r²)
 *
 * Shot noise approximated by Normal(0, g·I) for large photon counts.
 * Box-Muller transform for Gaussian sampling (no <random> overhead).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "phycam/types.hpp"

namespace phycam {

class SensorNoiseOperator {
public:
    double gain;           // calibrated shot-noise scale in normalised image units
    double read_noise_std; // read noise std in normalised [0,1] units
    bool   clip;           // clamp output to [0,1]
    unsigned int seed;

    explicit SensorNoiseOperator(double gain = 1.0,
                                  double read_noise_std = 0.005,
                                  bool clip = true,
                                  unsigned int seed = 12345)
        : gain(gain), read_noise_std(read_noise_std), clip(clip), seed(seed)
    {
        if (gain <= 0.0)
            throw std::invalid_argument("SensorNoiseOperator: gain must be > 0");
        if (read_noise_std < 0.0)
            throw std::invalid_argument("SensorNoiseOperator: read_noise_std must be >= 0");
    }

    /** Convenience constructor from ISO value. */
    static SensorNoiseOperator from_iso(int iso, int base_iso = 100,
                                        double base_read_noise = 0.002,
                                        double base_gain = 5e-5) {
        return SensorNoiseOperator(
                                   (static_cast<double>(iso) / base_iso) * base_gain,
                                   base_read_noise);
    }

    ImageBuffer& apply(ImageBuffer& buf) const {
        int N = static_cast<int>(buf.size());

        static constexpr double kPi = 3.14159265358979323846;

        // LCG state — one per call so results are reproducible
        unsigned int s = seed;
        auto lcg = [&s]() -> double {
            s = s * 1664525u + 1013904223u;
            return (s / 4294967296.0);  // [0, 1)
        };

        auto box_muller = [&]() -> std::pair<double, double> {
            double u1 = lcg() + 1e-12;
            double u2 = lcg();
            double mag = std::sqrt(-2.0 * std::log(u1));
            double ang = 2.0 * kPi * u2;
            return {mag * std::cos(ang), mag * std::sin(ang)};
        };

        float* ptr = buf.data.data();
        for (int i = 0; i < N; i += 2) {
            // Shot noise: σ_shot(I) = sqrt(gain * I)
            double sigma_shot0 = std::sqrt(gain * std::max(0.0, (double)ptr[i]));
            double sigma_shot1 = (i + 1 < N)
                ? std::sqrt(gain * std::max(0.0, (double)ptr[i + 1]))
                : 0.0;

            // Shot and read noise are independent Gaussian terms.
            auto [zs0, zs1] = box_muller();
            auto [zr0, zr1] = box_muller();

            float noisy0 = ptr[i]     + static_cast<float>(zs0 * sigma_shot0
                                       + zr0 * read_noise_std);
            float noisy1 = (i + 1 < N)
                ? ptr[i + 1] + static_cast<float>(zs1 * sigma_shot1
                               + zr1 * read_noise_std)
                : 0.0f;

            if (clip) {
                noisy0 = std::max(0.0f, std::min(1.0f, noisy0));
                noisy1 = std::max(0.0f, std::min(1.0f, noisy1));
            }
            ptr[i] = noisy0;
            if (i + 1 < N) ptr[i + 1] = noisy1;
        }
        return buf;
    }

    ImageBuffer apply_copy(const ImageBuffer& buf) const {
        ImageBuffer out = buf;
        apply(out);
        return out;
    }

    /** Approximate SNR in dB at a given normalised signal level. */
    double snr_db(double signal = 0.5) const {
        double variance = std::max(gain * signal + read_noise_std * read_noise_std, 1e-12);
        double snr = signal / std::sqrt(variance);
        return 20.0 * std::log10(std::max(snr, 1e-12));
    }
};

} // namespace phycam
