from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
import pywt

app = FastAPI(
    title="ECG MCP Server",
    description="Deterministic ECG tools for Watson Agents",
    version="1.0",
)

# -----------------------------
# INPUT SCHEMAS
# -----------------------------


class SignalInput(BaseModel):
    signal: list[float]
    fs: int = 300


class RRInput(BaseModel):
    rr_intervals: list[float]


# -----------------------------
# ROOT
# -----------------------------


@app.get("/")
def root():
    return {
        "name": "ECG MCP Server",
        "tools": [
            "signal_quality_index",
            "hrv_metrics",
            "ecg_preprocess_and_validate",
            "rpeak_and_hrv_stats",
            "fft_feature_summary",
            "wavelet_feature_summary",
        ],
    }


# -----------------------------
# TOOL 1: SIGNAL QUALITY INDEX
# -----------------------------


@app.post("/tools/signal_quality_index")
def signal_quality_index(data: SignalInput):
    signal = np.array(data.signal)
    fs = data.fs

    if len(signal) < fs:
        return {"quality": "poor", "reason": "Signal too short"}

    signal_power = np.mean(signal**2)
    noise_power = np.mean(np.diff(signal) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

    flatline_ratio = np.sum(np.abs(np.diff(signal)) < 1e-4) / len(signal)

    quality = "good"
    if snr < 10 or flatline_ratio > 0.2:
        quality = "poor"
    elif snr < 15:
        quality = "moderate"

    return {
        "snr_db": round(float(snr), 2),
        "flatline_ratio": round(float(flatline_ratio), 3),
        "signal_quality": quality,
    }


# -----------------------------
# TOOL 2: HRV METRICS
# -----------------------------


@app.post("/tools/hrv_metrics")
def hrv_metrics(data: RRInput):
    rr = np.array(data.rr_intervals)

    if len(rr) < 2:
        return {"error": "Not enough RR intervals"}

    diff_rr = np.diff(rr)

    return {
        "sdnn": round(float(np.std(rr)), 4),
        "rmssd": round(float(np.sqrt(np.mean(diff_rr**2))), 4),
        "pnn50": round(float(np.mean(np.abs(diff_rr) > 0.05)), 4),
        "rr_count": int(len(rr)),
    }


# -----------------------------
# TOOL 3: SIGNAL PROCESSING & VALIDATION
# -----------------------------


@app.post("/tools/ecg_preprocess_and_validate")
def ecg_preprocess_and_validate(data: SignalInput):
    signal = np.array(data.signal)
    fs = data.fs

    duration_sec = len(signal) / fs

    return {
        "length_samples": int(len(signal)),
        "duration_sec": round(float(duration_sec), 2),
        "mean": round(float(np.mean(signal)), 4),
        "std": round(float(np.std(signal)), 4),
        "min": round(float(np.min(signal)), 4),
        "max": round(float(np.max(signal)), 4),
    }


# -----------------------------
# TOOL 4: R-PEAK DETECTION + HRV STATS
# -----------------------------


@app.post("/tools/rpeak_and_hrv_stats")
def rpeak_and_hrv_stats(data: SignalInput):
    signal = np.array(data.signal)
    fs = data.fs

    threshold = np.mean(signal) + 1.2 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=int(0.3 * fs))

    if len(peaks) < 2:
        return {"error": "Insufficient R-peaks detected"}

    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / rr_intervals

    return {
        "rpeak_count": int(len(peaks)),
        "mean_bpm": round(float(np.mean(bpm)), 2),
        "std_bpm": round(float(np.std(bpm)), 2),
        "mean_rr": round(float(np.mean(rr_intervals)), 4),
        "rmssd": round(float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))), 4),
    }


# -----------------------------
# TOOL 5: FFT FEATURE SUMMARY
# -----------------------------


@app.post("/tools/fft_feature_summary")
def fft_feature_summary(data: SignalInput):
    signal = np.array(data.signal)
    fs = data.fs

    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[: N // 2]
    mag = 2.0 / N * np.abs(yf[: N // 2])

    band_mask = (xf >= 0.5) & (xf <= 40)
    freqs = xf[band_mask]
    mags = mag[band_mask]

    return {
        "dominant_frequency": round(float(freqs[np.argmax(mags)]), 2),
        "mean_magnitude": round(float(np.mean(mags)), 4),
        "band_energy": round(float(np.sum(mags**2)), 4),
        "low_freq_ratio": round(float(np.sum(mags[freqs < 5]) / np.sum(mags)), 4),
    }


# -----------------------------
# TOOL 6: WAVELET FEATURE SUMMARY
# -----------------------------


@app.post("/tools/wavelet_feature_summary")
def wavelet_feature_summary(data: SignalInput):
    signal = np.array(data.signal)

    coeffs = pywt.wavedec(signal, "db4", level=5)

    features = {}
    labels = ["A5", "D5", "D4", "D3", "D2", "D1"]

    for label, coeff in zip(labels, coeffs):
        features[f"{label}_energy"] = round(float(np.sum(coeff**2)), 4)
        features[f"{label}_entropy"] = round(
            float(-np.sum(coeff**2 * np.log(np.abs(coeff) + 1e-8))), 4
        )

    return features
