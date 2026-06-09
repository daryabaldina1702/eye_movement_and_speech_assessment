import numpy as np
import pandas as pd
from .config import CFG

def get_frame_width_from_scene_camera(scene_camera: dict) -> int | None:
    """
    Пытаемся вытащить ширину кадра без видео.
    В Neon scene_camera.json часто есть resolution/width/height (зависит от версии).
    Если не нашли — None.
    """
    if not scene_camera:
        return None
    # варианты
    if "resolution" in scene_camera and isinstance(scene_camera["resolution"], (list, tuple)) and len(scene_camera["resolution"]) == 2:
        return int(scene_camera["resolution"][0])
    if "width" in scene_camera:
        return int(scene_camera["width"])
    if "camera" in scene_camera and isinstance(scene_camera["camera"], dict):
        if "resolution" in scene_camera["camera"]:
            r = scene_camera["camera"]["resolution"]
            if isinstance(r, (list, tuple)) and len(r) == 2:
                return int(r[0])
    return None

def gaze_valid_mask(gaze: pd.DataFrame) -> pd.Series:
    """
    По ТЗ: исключать worn!=1 и blink id not null (если есть).
    """
    mask = pd.Series(True, index=gaze.index)
    if "worn" in gaze.columns:
        mask &= (gaze["worn"] == 1)
    if "blink id" in gaze.columns:
        mask &= gaze["blink id"].isna()
    return mask

def sustained_condition_time(ts: np.ndarray, cond: np.ndarray, min_hold_s: float) -> float | None:
    """
    Находит первое время, где cond=True удерживается непрерывно >= min_hold_s.
    Возвращает ts[start] или None.
    """
    if len(ts) == 0:
        return None
    i = 0
    n = len(ts)
    while i < n:
        if not cond[i]:
            i += 1
            continue
        j = i
        while j < n and cond[j]:
            j += 1
        # участок [i, j-1]
        if ts[j-1] - ts[i] >= min_hold_s:
            return float(ts[i])
        i = j
    return None

def compute_thr_px(rec: dict, cfg=CFG) -> int:
    w = get_frame_width_from_scene_camera(rec.get("scene_camera", {}))
    if w is None:
        return int(cfg["thr_px_fallback"])
    return int(max(cfg["thr_px_fallback"], cfg["thr_px_ratio"] * w))

def compute_jump_thr_px(rec: dict, cfg=CFG) -> int:
    w = get_frame_width_from_scene_camera(rec.get("scene_camera", {}))
    if w is None:
        return int(cfg["jump_thr_px_fallback"])
    return int(max(cfg["jump_thr_px_fallback"], cfg["jump_thr_px_ratio"] * w))

def gaze_based_rt(rec: dict, onset_s: float, cfg=CFG) -> dict:
    """
    По ТЗ:
      baseline_x/y = median gaze в [onset-0.20, onset-0.05]
      rt_s: первое t после onset+0.08..+0.70 где |gaze_x - baseline_x|>=thr_px удерживается >= hold_ms_reaction
      response_dir по медиане gaze_x в [rt_s, rt_s+0.12]
    """
    gaze = rec["gaze"]
    thr_px = compute_thr_px(rec, cfg)

    # baseline window
    base = gaze[(gaze["t_s"]>=onset_s-0.20) & (gaze["t_s"]<=onset_s-0.05)].copy()
    base = base[gaze_valid_mask(base)]
    if len(base) < 5:
        return {"qc_trial_valid": 0, "baseline_x": np.nan, "baseline_y": np.nan, "rt_s": None, "rt_found": 0,
                "rt_gaze_ms": np.nan, "response_dir": None, "thr_px": thr_px}

    baseline_x = float(base["gaze x [px]"].median())
    baseline_y = float(base["gaze y [px]"].median())

    # reaction search window
    w0, w1 = cfg["search_window_reaction"]
    seg = gaze[(gaze["t_s"]>=onset_s+w0) & (gaze["t_s"]<=onset_s+w1)].copy()
    seg = seg[gaze_valid_mask(seg)]
    if len(seg) < 10:
        return {"qc_trial_valid": 0, "baseline_x": baseline_x, "baseline_y": baseline_y, "rt_s": None, "rt_found": 0,
                "rt_gaze_ms": np.nan, "response_dir": None, "thr_px": thr_px}

    ts = seg["t_s"].values
    dx = (seg["gaze x [px]"].values - baseline_x)
    cond = np.abs(dx) >= thr_px

    rt_s = sustained_condition_time(ts, cond, cfg["hold_ms_reaction"]/1000.0)

    if rt_s is None:
        return {"qc_trial_valid": 1, "baseline_x": baseline_x, "baseline_y": baseline_y, "rt_s": None, "rt_found": 0,
                "rt_gaze_ms": np.nan, "response_dir": None, "thr_px": thr_px}

    rt_gaze_ms = float((rt_s - onset_s) * 1000.0)

    # response_dir
    resp = gaze[(gaze["t_s"]>=rt_s) & (gaze["t_s"]<=rt_s+0.12)].copy()
    resp = resp[gaze_valid_mask(resp)]
    if len(resp) < 5:
        response_dir = None
    else:
        dx_med = float(resp["gaze x [px]"].median() - baseline_x)
        response_dir = "RIGHT" if dx_med > 0 else "LEFT"

    return {"qc_trial_valid": 1, "baseline_x": baseline_x, "baseline_y": baseline_y,
            "rt_s": rt_s, "rt_found": 1, "rt_gaze_ms": rt_gaze_ms, "response_dir": response_dir, "thr_px": thr_px}

def target_acquired_time(rec: dict, onset_s: float, target_side: str, baseline_x: float, cfg=CFG) -> dict:
    """
    По ТЗ:
      target acquired если gaze попадает в сектор (LEFT/RIGHT) и удерживается >= hold_ms_target
      поиск в [onset, onset+1.5]
      time_to_target_ms = (t_target - onset)*1000
    """
    gaze = rec["gaze"]
    thr_px = compute_thr_px(rec, cfg)

    w0, w1 = cfg["search_window_target"]
    seg = gaze[(gaze["t_s"]>=onset_s+w0) & (gaze["t_s"]<=onset_s+w1)].copy()
    seg = seg[gaze_valid_mask(seg)]
    if len(seg) < 10 or (target_side not in ["LEFT","RIGHT"]) or np.isnan(baseline_x):
        return {"target_reached": 0, "t_target": None, "time_to_target_ms": np.nan}

    x = seg["gaze x [px]"].values
    ts = seg["t_s"].values

    if target_side == "LEFT":
        cond = x <= (baseline_x - thr_px)
    else:
        cond = x >= (baseline_x + thr_px)

    t_target = sustained_condition_time(ts, cond, cfg["hold_ms_target"]/1000.0)
    if t_target is None:
        return {"target_reached": 0, "t_target": None, "time_to_target_ms": np.nan}

    return {"target_reached": 1, "t_target": t_target, "time_to_target_ms": float((t_target - onset_s)*1000.0)}

def rt_saccade_reference(rec: dict, onset_s: float, cfg=CFG) -> dict:
    """
    По ТЗ: rt_saccade_ms только как reference (из saccades.csv).
    Берём первую саккаду со start в [onset+0.08, onset+0.70]
    """
    sac = rec["saccades"]
    w0, w1 = cfg["search_window_reaction"]
    cand = sac[(sac["t_start_s"]>=onset_s+w0) & (sac["t_start_s"]<=onset_s+w1)].sort_values("t_start_s")
    if cand.empty:
        return {"rt_saccade_found": 0, "rt_saccade_ms": np.nan}
    t = float(cand.iloc[0]["t_start_s"])
    return {"rt_saccade_found": 1, "rt_saccade_ms": float((t - onset_s)*1000.0)}