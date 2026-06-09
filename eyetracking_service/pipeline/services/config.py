from pathlib import Path

CFG = {
    "thr_px_ratio": 0.05,
    "thr_px_fallback": 60,
    "hold_ms_reaction": 40,
    "search_window_reaction": (0.08, 0.70),

    "hold_ms_target": 80,
    "search_window_target": (0.0, 1.5),

    "express_thr_ms": 120,

    "jump_thr_px_ratio": 0.03,
    "jump_thr_px_fallback": 40,

    "qc_worn_min": 0.8,
    "qc_gaze_rate_min_hz": 50,
    "qc_imu_gyro_rms_max": 50.0,
}