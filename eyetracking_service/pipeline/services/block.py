import pandas as pd
import numpy as np 

def summarize_blocks(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    summarize_blocks(trial_metrics) -> df block_summary
    По ТЗ:
      n_trials
      rt_gaze_ms_mean/median/std
      direction_error_rate
      express_like_rate
      target_reached_rate
      time_to_target_ms_mean
    + gap_overlap_delta_rt_ms = median(RT_OVERLAP) - median(RT_GAP)
    """
    if trial_metrics is None or trial_metrics.empty:
        return pd.DataFrame()

    df = trial_metrics.copy()

    # Для стандартных блоков используем rt_gaze_ms
    group_cols = ["recording_id","block"]
    rows = []

    for (rid, b), g in df.groupby(group_cols):
        n_trials = len(g)
        rt = g["rt_gaze_ms"] if "rt_gaze_ms" in g.columns else pd.Series(dtype=float)

        row = {
            "recording_id": rid,
            "block": b,
            "n_trials": n_trials,
            "rt_gaze_ms_mean": float(rt.mean()) if rt.notna().any() else np.nan,
            "rt_gaze_ms_median": float(rt.median()) if rt.notna().any() else np.nan,
            "rt_gaze_ms_std": float(rt.std()) if rt.notna().any() else np.nan,
            "direction_error_rate": float(g["direction_error"].mean()) if "direction_error" in g.columns and g["direction_error"].notna().any() else np.nan,
            "express_like_rate": float(g["express_like"].mean()) if "express_like" in g.columns and g["express_like"].notna().any() else np.nan,
            "target_reached_rate": float(g["target_reached"].mean()) if "target_reached" in g.columns and g["target_reached"].notna().any() else np.nan,
            "time_to_target_ms_mean": float(g["time_to_target_ms"].mean()) if "time_to_target_ms" in g.columns and g["time_to_target_ms"].notna().any() else np.nan,
        }

        # ANTISACCADE: полезно отдельно хранить rt_gaze_ms_mean по цветам можно потом
        rows.append(row)

    out = pd.DataFrame(rows)

    # gap_overlap_delta_rt_ms по recording_id
    gap = out[out["block"]=="GAP"][["recording_id","rt_gaze_ms_median"]].rename(columns={"rt_gaze_ms_median":"rt_gap_median"})
    ov  = out[out["block"]=="OVERLAP"][["recording_id","rt_gaze_ms_median"]].rename(columns={"rt_gaze_ms_median":"rt_overlap_median"})
    go = gap.merge(ov, on="recording_id", how="outer")
    go["gap_overlap_delta_rt_ms"] = go["rt_overlap_median"] - go["rt_gap_median"]

    out = out.merge(go[["recording_id","gap_overlap_delta_rt_ms"]], on="recording_id", how="left")
    return out.sort_values(["recording_id","block"]).reset_index(drop=True)