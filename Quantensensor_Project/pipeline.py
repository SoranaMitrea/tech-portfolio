# src/pipeline.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config (keep simple for now)
# -----------------------------
DEFAULT_CFG = {
    "fs_hz": 1000.0,
    "preprocess": {
        "detrend": True,
        "lowpass_hz": 10.0, # Signal ~2 Hz, LPF 10 Hz ist ok
        "filter_order": 4
    },
    "monitor": {
        "window_sec": 0.05, # Rolling window für z-score
        "spike": {
            "enabled": True,
            "z_thresh": 5.0,
            "min_duration_sec": 0.0 # mind. 10 ms
        }
    },
    "export": {
        "out_dir": "data/reports/run_001",
        "save_csv": True,
        "save_json": True,
        "save_plots": True
    }
}


# -----------------------------
# Helpers
# -----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "B_field_T"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")
    return df


def _try_lowpass_scipy(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> Optional[np.ndarray]:
    """
    Returns filtered signal if scipy available, else None.
    """
    try:
        from scipy.signal import butter, filtfilt # type: ignore
    except Exception:
        return None

    wn = cutoff_hz / (fs / 2.0)
    wn = min(max(wn, 1e-6), 0.999999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def preprocess_signal(b: np.ndarray, t: np.ndarray, fs_hz: float, cfg_pre: dict) -> np.ndarray:
    x = b.astype(float)

    # detrend (linear)
    if cfg_pre.get("detrend", True):
        # Linear least squares detrend
        tt = t - t[0]
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, offset = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (slope * tt + offset)

    # lowpass
    lp = float(cfg_pre.get("lowpass_hz", 0.0) or 0.0)
    if lp > 0:
        order = int(cfg_pre.get("filter_order", 4))
        y = _try_lowpass_scipy(x, fs_hz, lp, order)
        if y is not None:
            x = y
        else:
            # Fallback: simple moving average (not as good, but no extra deps)
            win = max(3, int(fs_hz / lp)) # rough heuristic
            win = min(win, max(3, len(x)//50))
            kernel = np.ones(win) / win
            x = np.convolve(x, kernel, mode="same")

    return x


def compute_kpis(t: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    sigma = float(np.std(b))
    p2p = float(np.max(b) - np.min(b))

    tt = t - t[0]
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, offset = np.linalg.lstsq(A, b, rcond=None)[0]
    drift_rate = float(slope) # T/s

    snr_proxy = float(p2p / (2.0 * sigma + 1e-12))

    return {
        "sigma_T": sigma,
        "peak_to_peak_T": p2p,
        "drift_rate_T_per_s": drift_rate,
        "snr_proxy": snr_proxy
    }


def rolling_zscore(x: np.ndarray, win: int) -> np.ndarray:
    z = np.zeros_like(x, dtype=float)
    if win < 5:
        win = 5
    for i in range(win, len(x)):
        seg = x[i - win:i]
        mu = seg.mean()
        sd = seg.std() + 1e-12
        z[i] = (x[i] - mu) / sd
    return z


def detect_spikes(t: np.ndarray, b_raw: np.ndarray, b: np.ndarray, fs_hz: float, cfg_monitor: dict) -> Dict:
    spike_cfg = cfg_monitor.get("spike", {})
    if not spike_cfg.get("enabled", True):
        return {"spike_count": 0, "spike_segments": []}

    win_sec = float(cfg_monitor.get("window_sec", 0.2))
    win = max(5, int(win_sec * fs_hz))

    z_thresh = float(spike_cfg.get("z_thresh", 2.0))
    min_dur = float(spike_cfg.get("min_duration_sec", 0.00))
    min_len = max(1, int(min_dur * fs_hz))

    db = np.diff(b_raw, prepend=b_raw[0])   # Änderung pro Sample, Step-Kanten werden sichtbar
    z = rolling_zscore(db, win)
    idx = np.where(np.abs(z) > z_thresh)[0]

    segments: List[Tuple[float, float]] = []
    if len(idx) > 0:
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
            else:
                if (prev - start + 1) >= min_len:
                    segments.append((float(t[start]), float(t[prev])))
                start = k
                prev = k
        if (prev - start + 1) >= min_len:
            segments.append((float(t[start]), float(t[prev])))

    return {
        "spike_count": len(segments),
        "spike_segments": segments
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_overview(out_dir: Path, t: np.ndarray, b_raw: np.ndarray, b_proc: np.ndarray, title: str) -> None:
    plt.figure()
    plt.plot(t, b_raw, label="B_raw")
    plt.plot(t, b_proc, label="B_proc")
    plt.xlabel("time [s]")
    plt.ylabel("B [T]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_overview.png", dpi=150)

def plot_spike_zoom(out_dir: Path, t: np.ndarray, b: np.ndarray, segments: List[Tuple[float, float]]) -> None:
    # Zoom auf den ersten Spike
    if not segments:
        return
    t0, t1 = segments[0]
    # bisschen Rand
    pad = 0.2
    lo = max(t[0], t0 - pad)
    hi = min(t[-1], t1 + pad)

    mask = (t >= lo) & (t <= hi)
    plt.figure()
    plt.plot(t[mask], b[mask])
    plt.xlabel("time [s]")
    plt.ylabel("B [T]")
    plt.title(f"Spike zoom: {t0:.3f}s - {t1:.3f}s")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_spike_zoom.png", dpi=150)

# -----------------------------
# Main
# -----------------------------
def run(input_csv: Path, cfg: dict = DEFAULT_CFG) -> None:
    df = load_csv(input_csv)
    t = df["time"].to_numpy(dtype=float)
    b_raw = df["B_field_T"].to_numpy(dtype=float)

    db_raw = np.diff(b_raw, prepend=b_raw[0])
    i = int(np.argmax(np.abs(db_raw)))
    print("DEBUG max|db_raw|:", np.abs(db_raw[i]), "at idx", i, "t", t[i], "s")

    fs = float(cfg["fs_hz"])
    b_proc = preprocess_signal(b_raw, t, fs, cfg["preprocess"])

    kpis = compute_kpis(t, b_proc)

# extra Monitor-Signal: weniger glätten, damit Step-Kanten sichtbar bleiben cfg_mon_pre = dict(cfg["preprocess"])
    cfg_mon_pre = dict(cfg["preprocess"])
    cfg_mon_pre["lowpass_hz"] = 50.0   # 30..100 Hz testen
    b_mon = preprocess_signal(b_raw, t, fs, cfg_mon_pre)

    faults = detect_spikes(t, b_raw, b_mon, fs, cfg["monitor"])


    out_dir = Path(cfg["export"]["out_dir"])
    ensure_dir(out_dir)

# export files
    if cfg["export"].get("save_csv", True):
        df_out = df.copy()
        df_out["B_proc_T"] = b_proc
        df_out.to_csv(out_dir / "processed.csv", index=False)

    if cfg["export"].get("save_json", True):
        save_json(out_dir / "kpis.json", kpis)
        save_json(out_dir / "faults.json", faults)

    if cfg["export"].get("save_plots", True):
        plot_overview(out_dir, t, b_raw, b_proc, title=f"Overview: {input_csv.name}")
        plot_spike_zoom(out_dir, t, b_proc, faults.get("spike_segments", []))

    print("=== PIPELINE DONE ===")
    print(f"Input: {input_csv}")
    print(f"Out: {out_dir.resolve()}")
    print("KPIs:", kpis)
    print("Faults:", faults)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <path_to_csv>")
        print("Example: python src/pipeline.py data/raw/signal_fault.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()
    run(input_path)

