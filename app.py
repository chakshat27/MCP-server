from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

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
# ROOT (Optional but helpful)
# -----------------------------


@app.get("/")
def root():
    return {"name": "ECG MCP Server", "tools": ["signal_quality_index", "hrv_metrics"]}


# -----------------------------
# TOOL 1: SIGNAL QUALITY INDEX
# -----------------------------


@app.post("/tools/signal_quality_index")
def signal_quality_index(data: SignalInput):
    signal = np.array(data.signal)
    fs = data.fs

    if len(signal) < fs:
        return {"quality": "poor", "reason": "Signal too short"}

    # Signal power
    signal_power = np.mean(signal**2)

    # High-frequency noise estimate (simple diff-based proxy)
    noise_power = np.mean(np.diff(signal) ** 2)

    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

    # Flatline detection
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

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    pnn50 = np.mean(np.abs(diff_rr) > 0.05)

    return {
        "sdnn": round(float(sdnn), 4),
        "rmssd": round(float(rmssd), 4),
        "pnn50": round(float(pnn50), 4),
        "rr_count": int(len(rr)),
    }
