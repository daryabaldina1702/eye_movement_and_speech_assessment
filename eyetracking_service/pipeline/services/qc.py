import numpy as np
import pandas as pd
from .config import CFG

def estimate_gaze_rate_hz(gaze_t: pd.Series) -> float:
    ts = gaze_t.dropna().values
    if len(ts) < 5:
        return np.nan
    dt = np.diff(ts)
    dt = dt[(dt > 0) & (dt < np.quantile(dt, 0.99))]
    if len(dt) == 0:
        return np.nan
    return float(1.0 / np.median(dt))

def imu_gyro_rms(imu: pd.DataFrame) -> float:
    if imu is None or imu.empty:
        return np.nan
    gyro_cols = [c for c in imu.columns if c.lower().startswith("gyro")]
    if not gyro_cols:
        return np.nan
    arr = imu[gyro_cols].fillna(0).values
    return float(np.sqrt(np.mean(arr**2)))

def compute_qc(rec: dict, cfg=CFG) -> dict:
    """
    compute_qc(rec) -> row dict для subject_qc.csv
    По ТЗ: обязательные поля + qc flags.
    """
    gaze = rec["gaze"]
    fix = rec["fixations"]
    sac = rec["saccades"]
    bl  = rec["blinks"]

    recording_id = rec["recording_id"]

    # duration_s
    if rec["info"].get("duration"):
        duration_s = float(rec["info"]["duration"]) / 1e9
    else:
        duration_s = float(gaze["t_s"].max()) if not gaze.empty else np.nan

    gaze_rate_hz = estimate_gaze_rate_hz(gaze["t_s"]) if not gaze.empty else np.nan
    worn_ratio = float(gaze["worn"].mean()) if (not gaze.empty and "worn" in gaze.columns) else np.nan
    n_gaze_samples = int(len(gaze))

    blink_rate_per_min = (len(bl) / (duration_s/60)) if (duration_s and not bl.empty) else np.nan

    fix_dur_med = float(fix["duration [ms]"].median()) if (not fix.empty and "duration [ms]" in fix.columns) else np.nan
    fix_dur_mean = float(fix["duration [ms]"].mean()) if (not fix.empty and "duration [ms]" in fix.columns) else np.nan

    n_saccades = int(len(sac))
    sacc_amp_deg_mean = float(sac["amplitude [deg]"].mean()) if (not sac.empty and "amplitude [deg]" in sac.columns) else np.nan
    sacc_peak_px_s_mean = float(sac["peak velocity [px/s]"].mean()) if (not sac.empty and "peak velocity [px/s]" in sac.columns) else np.nan

    imu_rms = imu_gyro_rms(rec["imu"])

    qc_flag_low_worn = int((not np.isnan(worn_ratio)) and (worn_ratio < cfg["qc_worn_min"]))
    qc_flag_low_rate = int((not np.isnan(gaze_rate_hz)) and (gaze_rate_hz < cfg["qc_gaze_rate_min_hz"]))
    qc_flag_high_motion = int((not np.isnan(imu_rms)) and (imu_rms > cfg["qc_imu_gyro_rms_max"]))

    return {
        "recording_id": recording_id,
        "duration_s": duration_s,
        "gaze_rate_hz": gaze_rate_hz,
        "worn_ratio": worn_ratio,
        "n_gaze_samples": n_gaze_samples,
        "blink_rate_per_min": blink_rate_per_min,
        "n_fixations": int(len(fix)),
        "fix_dur_ms_median": fix_dur_med,
        "fix_dur_ms_mean": fix_dur_mean,
        "n_saccades": n_saccades,
        "sacc_amp_deg_mean": sacc_amp_deg_mean,
        "sacc_peak_px_s_mean": sacc_peak_px_s_mean,
        "imu_gyro_rms": imu_rms,
        "qc_flag_low_worn": qc_flag_low_worn,
        "qc_flag_low_rate": qc_flag_low_rate,
        "qc_flag_high_motion": qc_flag_high_motion,
    }