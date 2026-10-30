"""Tests for device calibration and measured peaks."""

from stackkernels._opt.calibrate import calibrate_device, measured_peaks
from tests.fixtures.profiles import CPU_PROFILE, T4_PROFILE


def test_measured_peaks_uncalibrated():
    peaks = measured_peaks(T4_PROFILE)
    assert peaks["bandwidth_gbps"] == 320.0
    assert peaks["tflops_fp16"] == 65.0
    assert peaks["tflops_fp32"] == 8.1
    assert peaks["calibrated"] is False


def test_calibrate_device_cpu():
    res = calibrate_device(CPU_PROFILE, force=True)
    assert res["calibrated"] is True
    assert res["bandwidth_gbps"] > 0.0

    peaks = measured_peaks(CPU_PROFILE)
    assert peaks["calibrated"] is True
