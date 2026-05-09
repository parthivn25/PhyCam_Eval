"""
Tests that the compiled C++ backend is available when running the backend suite.
"""

from phycam_eval.degradations import hdr, noise, optical


def test_cpp_backend_is_available():
    assert optical._CPP, "DefocusOperator is using the Python fallback; run ./scripts/build.sh"
    assert hdr._CPP, "HDRCompressionOperator is using the Python fallback; run ./scripts/build.sh"
    assert noise._CPP, "SensorNoiseOperator is using the Python fallback; run ./scripts/build.sh"
