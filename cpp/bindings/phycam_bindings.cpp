/**
 * phycam_bindings.cpp — pybind11 Python bindings for phycam-eval C++ operators.
 *
 * Exposes all degradation operators as Python classes that accept and return
 * numpy float32 arrays in CHW layout — compatible with PyTorch convention.
 *
 * Usage from Python:
 *   import phycam_cpp
 *   arr = image.numpy()  # (C, H, W) float32
 *   out = phycam_cpp.DefocusOperator(alpha=1.5).apply(arr)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <cstring>

#include "phycam/types.hpp"
#include "phycam/optical.hpp"
#include "phycam/hdr.hpp"
#include "phycam/noise.hpp"

namespace py = pybind11;
using namespace phycam;

// ---- helpers ---------------------------------------------------------------

/** Convert a (C,H,W) numpy float32 array → ImageBuffer. */
ImageBuffer from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto info = arr.request();
    if (info.ndim != 3)
        throw std::invalid_argument("Expected (C, H, W) array");
    int C = static_cast<int>(info.shape[0]);
    int H = static_cast<int>(info.shape[1]);
    int W = static_cast<int>(info.shape[2]);
    ImageBuffer buf(C, H, W);
    std::memcpy(buf.data.data(), info.ptr, C * H * W * sizeof(float));
    return buf;
}

/** Convert ImageBuffer → (C,H,W) numpy float32 array. */
py::array_t<float> to_numpy(const ImageBuffer& buf) {
    py::array_t<float> out({buf.channels, buf.height, buf.width});
    std::memcpy(out.mutable_data(), buf.data.data(),
                buf.size() * sizeof(float));
    return out;
}

/** Generic apply: operator→ apply_copy → numpy. */
template<typename Op>
py::array_t<float> apply_op(
    const Op& op,
    py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = from_numpy(arr);
    op.apply(buf);
    return to_numpy(buf);
}

// ---- module ----------------------------------------------------------------

PYBIND11_MODULE(phycam_cpp, m) {
    m.doc() = "PhyCam-Eval C++ degradation operators (FFTW3-backed)";

    // ----------------------------------------------------------------
    // DefocusOperator
    // ----------------------------------------------------------------
    py::class_<DefocusOperator>(m, "DefocusOperator",
        R"(
        Defocus via quadratic pupil phase (PST/Fourier optics grounded).

        A_phi(I) = F^{-1} { F(I) * exp(j * alpha * rho^2) }

        Parameters
        ----------
        alpha : float >= 0
            Phase strength.  0 = identity.  ~pi = visible blur.
        normalize_freq : bool
            Normalise rho to [0,1] before applying phi (default True).
        )")
        .def(py::init<double, bool>(),
             py::arg("alpha") = 1.0, py::arg("normalize_freq") = true)
        .def_readwrite("alpha",         &DefocusOperator::alpha)
        .def_readwrite("normalize_freq",&DefocusOperator::normalize_freq)
        .def("apply", [](const DefocusOperator& op,
                         py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return apply_op(op, arr);
        }, py::arg("image"),
           "Apply defocus to a (C,H,W) float32 numpy array. Returns degraded array.")
        .def("phase_mask", [](const DefocusOperator& op, int H, int W) {
            auto phi = op.phase_mask(H, W);
            py::array_t<double> out({H, W});
            std::memcpy(out.mutable_data(), phi.data(), H * W * sizeof(double));
            return out;
        }, py::arg("H"), py::arg("W"),
           "Return the (H,W) phase mask phi(rho) as a numpy array.")
        .def("otf_magnitude", [](const DefocusOperator& op, int H, int W) {
            auto otf = op.otf_magnitude(H, W);
            py::array_t<double> out({H, W});
            std::memcpy(out.mutable_data(), otf.data(), H * W * sizeof(double));
            return out;
        }, py::arg("H"), py::arg("W"),
           "Return |OTF(f)| = MTF for validation. Phase-only → all 1s.")
        .def("__repr__", [](const DefocusOperator& op) {
            return "DefocusOperator(alpha=" + std::to_string(op.alpha) +
                   ", normalize_freq=" + std::string(op.normalize_freq ? "true" : "false") + ")";
        });

    // ----------------------------------------------------------------
    // AstigmatismOperator
    // ----------------------------------------------------------------
    py::class_<AstigmatismOperator>(m, "AstigmatismOperator",
        R"(
        Astigmatism via PAGE-derived axis-dependent quadratic phase.

        phi(rho, theta) = alpha * cos^2(theta - theta_axis) * rho^2
        Equivalent to Zernike Z2+2 (oblique astigmatism).

        Parameters
        ----------
        alpha      : float — maximum phase strength
        theta_axis : float — blur axis angle in radians (0 = horizontal)
        )")
        .def(py::init<double, double>(),
             py::arg("alpha") = 1.0, py::arg("theta_axis") = 0.0)
        .def_readwrite("alpha",      &AstigmatismOperator::alpha)
        .def_readwrite("theta_axis", &AstigmatismOperator::theta_axis)
        .def("apply", [](const AstigmatismOperator& op,
                         py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return apply_op(op, arr);
        }, py::arg("image"))
        .def("phase_mask", [](const AstigmatismOperator& op, int H, int W) {
            auto phi = op.phase_mask(H, W);
            py::array_t<double> out({H, W});
            std::memcpy(out.mutable_data(), phi.data(), H * W * sizeof(double));
            return out;
        }, py::arg("H"), py::arg("W"))
        .def("__repr__", [](const AstigmatismOperator& op) {
            return "AstigmatismOperator(alpha=" + std::to_string(op.alpha) +
                   ", theta_axis=" + std::to_string(op.theta_axis) + ")";
        });

    // ----------------------------------------------------------------
    // LowLightOperator
    // ----------------------------------------------------------------
    py::class_<LowLightOperator>(m, "LowLightOperator",
        R"(
        Low-light degradation: VEViD-inspired photon-starvation model.

        H_ll(rho) = 1 / (1 + (rho / sqrt(light_level))^{2*order})
        followed by shot noise: sigma = 0.05 / sqrt(light_level).

        Parameters
        ----------
        light_level : float in (0,1]  — 1.0=bright, 0.05=very dark
        order       : int             — Butterworth order (default 2)
        seed        : int             — RNG seed for reproducibility
        )")
        .def(py::init<double, int, unsigned int>(),
             py::arg("light_level") = 0.5,
             py::arg("order") = 2,
             py::arg("seed") = 42)
        .def_readwrite("light_level", &LowLightOperator::light_level)
        .def_readwrite("order",       &LowLightOperator::order)
        .def("apply", [](const LowLightOperator& op,
                         py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return apply_op(op, arr);
        }, py::arg("image"))
        .def("__repr__", [](const LowLightOperator& op) {
            return "LowLightOperator(light_level=" + std::to_string(op.light_level) +
                   ", order=" + std::to_string(op.order) +
                   ", seed=" + std::to_string(op.rng_seed) + ")";
        });

    // ----------------------------------------------------------------
    // HDRCompressionOperator
    // ----------------------------------------------------------------
    py::class_<HDRCompressionOperator>(m, "HDRCompressionOperator",
        R"(
        ODRC-inspired HDR dynamic range compression.

        Q_beta(I) = F^{-1} { sign(F(I)) * |F(I)|^beta }

        Parameters
        ----------
        beta        : float > 0   — 1.0=identity, <1=compress, >1=expand
        per_channel : bool        — apply per-channel (default True)
        eps         : float       — avoid divide-by-zero (default 1e-10)
        )")
        .def(py::init<double, bool, double>(),
             py::arg("beta") = 0.5,
             py::arg("per_channel") = true,
             py::arg("eps") = 1e-10)
        .def_readwrite("beta",        &HDRCompressionOperator::beta)
        .def_readwrite("per_channel", &HDRCompressionOperator::per_channel)
        .def("apply", [](const HDRCompressionOperator& op,
                         py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return apply_op(op, arr);
        }, py::arg("image"))
        .def("compression_ratio_db", &HDRCompressionOperator::compression_ratio_db,
             "Return 20*log10(beta) in dB (negative = compression).")
        .def("__repr__", [](const HDRCompressionOperator& op) {
            return "HDRCompressionOperator(beta=" + std::to_string(op.beta) +
                   ", per_channel=" + std::string(op.per_channel ? "true" : "false") +
                   ", eps=" + std::to_string(op.eps) + ")";
        });

    // ----------------------------------------------------------------
    // SensorNoiseOperator
    // ----------------------------------------------------------------
    py::class_<SensorNoiseOperator>(m, "SensorNoiseOperator",
        R"(
        Mixed Poisson-Gaussian sensor noise: N(I) = Poisson(I/g)*g + Normal(0, sigma_r^2).

        Parameters
        ----------
        gain           : float > 0  — calibrated shot-noise scale
        read_noise_std : float >= 0 — read noise std in normalised units
        clip           : bool       — clamp output to [0,1] (default True)
        seed           : int        — RNG seed
        )")
        .def(py::init<double, double, bool, unsigned int>(),
             py::arg("gain") = 1.0,
             py::arg("read_noise_std") = 0.005,
             py::arg("clip") = true,
             py::arg("seed") = 12345)
        .def_readwrite("gain",           &SensorNoiseOperator::gain)
        .def_readwrite("read_noise_std", &SensorNoiseOperator::read_noise_std)
        .def_static("from_iso", &SensorNoiseOperator::from_iso,
                    py::arg("iso"), py::arg("base_iso") = 100,
                    py::arg("base_read_noise") = 0.002,
                    py::arg("base_gain") = 5e-5)
        .def("apply", [](const SensorNoiseOperator& op,
                         py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            return apply_op(op, arr);
        }, py::arg("image"))
        .def("snr_db", &SensorNoiseOperator::snr_db,
             py::arg("signal") = 0.5,
             "Approximate SNR in dB at a given normalised signal level.")
        .def("__repr__", [](const SensorNoiseOperator& op) {
            return "SensorNoiseOperator(gain=" + std::to_string(op.gain) +
                   ", read_noise_std=" + std::to_string(op.read_noise_std) +
                   ", clip=" + std::string(op.clip ? "true" : "false") +
                   ", seed=" + std::to_string(op.seed) + ")";
        });

}
