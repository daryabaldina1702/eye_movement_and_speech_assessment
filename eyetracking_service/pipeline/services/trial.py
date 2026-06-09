import pandas as pd
import numpy as np
from .help_func import gaze_based_rt, gaze_valid_mask, compute_jump_thr_px, rt_saccade_reference, target_acquired_time
from .config import CFG

def trial_window_from_parsed(parsed_block: pd.DataFrame, trial_id: int, onset_s: float) -> tuple[float, float]:
    """
    По ТЗ:
      окно trial для фиксаций:
        [step1_start, step2_end] если есть,
        иначе [onset-0.5, onset+1.5]
    """
    t0 = onset_s - 0.5
    t1 = onset_s + 1.5

    sub = parsed_block[parsed_block["trial_id"]==trial_id]
    # step1 start
    s1 = sub[(sub["step"]==1) & (sub["edge"]=="start")]["t_s"]
    # step2 end
    e2 = sub[(sub["step"]==2) & (sub["edge"]=="end")]["t_s"]

    if not s1.empty:
        t0 = float(s1.min())
    if not e2.empty:
        t1 = float(e2.max())
    return t0, t1

def fixation_trial_stats(rec: dict, t0: float, t1: float) -> dict:
    """
    По ТЗ:
      fix_dur_ms_mean_trial / median_trial
    """
    fix = rec["fixations"]
    if fix is None or fix.empty or "duration [ms]" not in fix.columns:
        return {"fix_dur_ms_mean_trial": np.nan, "fix_dur_ms_median_trial": np.nan}

    # фиксация попадает в окно, если начало/конец пересекаются с [t0,t1]
    seg = fix[(fix["t_end_s"]>=t0) & (fix["t_start_s"]<=t1)]
    if seg.empty:
        return {"fix_dur_ms_mean_trial": np.nan, "fix_dur_ms_median_trial": np.nan}
    return {
        "fix_dur_ms_mean_trial": float(seg["duration [ms]"].mean()),
        "fix_dur_ms_median_trial": float(seg["duration [ms]"].median())
    }

def count_large_jumps(gaze: pd.DataFrame, t0: float, t1: float, jump_thr_px: float) -> int:
    """
    По ТЗ:
      n_large_jumps_to_target — proxy по gaze:
      |Δgaze_x| между соседними точками > jump_thr_px (исключая blink/worn)
    """
    seg = gaze[(gaze["t_s"]>=t0) & (gaze["t_s"]<=t1)].copy()
    seg = seg[gaze_valid_mask(seg)]
    if len(seg) < 3:
        return 0
    dx = np.diff(seg["gaze x [px]"].values)
    return int(np.sum(np.abs(dx) > jump_thr_px))

def compute_trial_metrics_for_block(rec: dict, parsed_events: pd.DataFrame, block: str, cfg=CFG) -> pd.DataFrame:
    """
    compute_trial_metrics(rec, parsed_events, block) -> df(trial_metrics)
    По ТЗ:
      - PREDICTION/GAP/OVERLAP: target_side из step=2 start, side LEFT/RIGHT
      - DECISION: onset=min(step2 start), correct_side из step3 CORRECT
      - ANTISACCADE: cue_color на step=2 start (RED/GREEN), без direction_error
    """
    pe = parsed_events[parsed_events["block"]==block].copy()
    if pe.empty:
        return pd.DataFrame()

    rows = []
    gaze = rec["gaze"]
    jump_thr_px = compute_jump_thr_px(rec, cfg)

    # trials list
    for trial_id in sorted(pe["trial_id"].unique()):
        trial = pe[pe["trial_id"]==trial_id].copy()

        base_row = {
            "recording_id": rec["recording_id"],
            "block": block,
            "trial_id": int(trial_id),
            "onset_s": np.nan,
            "target_side": None,
            "qc_trial_valid": 0,
        }

        if block in ["PREDICTION","GAP","OVERLAP"]:
            onset_candidates = trial[(trial["step"]==2) & (trial["edge"]=="start") & (trial["side"].isin(["LEFT","RIGHT"]))][["t_s","side"]]
            if onset_candidates.empty:
                rows.append(base_row)
                continue

            onset_s = float(onset_candidates["t_s"].min())
            target_side = onset_candidates.sort_values("t_s").iloc[0]["side"]

            base_row["onset_s"] = onset_s
            base_row["target_side"] = target_side

            # RT (gaze-based)
            rt = gaze_based_rt(rec, onset_s, cfg)
            base_row["qc_trial_valid"] = int(rt["qc_trial_valid"])
            base_row["rt_found"] = int(rt["rt_found"])
            base_row["rt_gaze_ms"] = rt["rt_gaze_ms"]
            base_row["response_dir"] = rt["response_dir"]

            # saccade reference
            sref = rt_saccade_reference(rec, onset_s, cfg)
            base_row.update(sref)

            # direction error
            if base_row["rt_found"] == 1 and base_row["response_dir"] in ["LEFT","RIGHT"]:
                base_row["direction_error"] = int(base_row["response_dir"] != target_side)
            else:
                base_row["direction_error"] = np.nan

            # express-like
            base_row["express_like"] = int((base_row["rt_found"]==1) and (base_row["rt_gaze_ms"] < cfg["express_thr_ms"]))

            # target acquired
            ta = target_acquired_time(rec, onset_s, target_side, rt["baseline_x"], cfg)
            base_row["target_reached"] = int(ta["target_reached"])
            base_row["time_to_target_ms"] = ta["time_to_target_ms"]

            # post_response_dx_px + accuracy_sector
            if rt["rt_s"] is not None:
                post = gaze[(gaze["t_s"]>=rt["rt_s"]+0.15) & (gaze["t_s"]<=rt["rt_s"]+0.35)].copy()
                post = post[gaze_valid_mask(post)]
                if len(post) >= 5:
                    post_dx = float(post["gaze x [px]"].median() - rt["baseline_x"])
                else:
                    post_dx = np.nan
            else:
                post_dx = np.nan

            base_row["post_response_dx_px"] = post_dx
            if not np.isnan(post_dx) and target_side in ["LEFT","RIGHT"]:
                base_row["accuracy_sector"] = int((post_dx > 0 and target_side=="RIGHT") or (post_dx < 0 and target_side=="LEFT"))
            else:
                base_row["accuracy_sector"] = np.nan

            # фиксации в окне trial
            t0, t1 = trial_window_from_parsed(pe, trial_id, onset_s)
            base_row.update(fixation_trial_stats(rec, t0, t1))

            # large jumps до цели
            if ta["t_target"] is not None:
                n_jumps = count_large_jumps(gaze, onset_s, ta["t_target"], jump_thr_px)
            else:
                n_jumps = count_large_jumps(gaze, onset_s, onset_s + cfg["search_window_target"][1], jump_thr_px)

            base_row["n_large_jumps_to_target"] = n_jumps
            base_row["n_steps_to_target"] = n_jumps  # proxy по ТЗ

            # доля времени на достижение цели
            # trial_duration_ms: step2_end - onset, иначе 2000ms
            step2_end = trial[(trial["step"]==2) & (trial["edge"]=="end")]["t_s"]
            if not step2_end.empty:
                trial_dur_ms = float((step2_end.max() - onset_s) * 1000.0)
            else:
                trial_dur_ms = 2000.0

            base_row["trial_duration_ms"] = trial_dur_ms
            if base_row["target_reached"] == 1 and not np.isnan(base_row["time_to_target_ms"]) and trial_dur_ms > 0:
                base_row["time_to_target_share"] = float(base_row["time_to_target_ms"] / trial_dur_ms)
            else:
                base_row["time_to_target_share"] = np.nan

            rows.append(base_row)
            continue

        if block == "DECISION":
            onset_candidates = trial[(trial["step"]==2) & (trial["edge"]=="start")][["t_s"]]
            if onset_candidates.empty:
                rows.append(base_row)
                continue

            onset_s = float(onset_candidates["t_s"].min())
            base_row["onset_s"] = onset_s

            # correct_side из step=3 CORRECT
            corr = trial[(trial["step"]==3) & (trial["correctness"]=="CORRECT") & (trial["edge"]=="start")]["side"]
            correct_side = corr.iloc[0] if not corr.empty else None

            # RT по gaze (decision_rt_gaze_ms) + choice_side
            rt = gaze_based_rt(rec, onset_s, cfg)
            base_row["qc_trial_valid"] = int(rt["qc_trial_valid"])
            base_row["decision_rt_gaze_ms"] = rt["rt_gaze_ms"]
            base_row["choice_side"] = rt["response_dir"]
            base_row["correct_side"] = correct_side

            if (base_row["choice_side"] in ["LEFT","RIGHT"]) and (correct_side in ["LEFT","RIGHT"]):
                base_row["decision_correct"] = int(base_row["choice_side"] == correct_side)
            else:
                base_row["decision_correct"] = np.nan

            # фиксации trial
            t0, t1 = trial_window_from_parsed(pe, trial_id, onset_s)
            base_row.update(fixation_trial_stats(rec, t0, t1))

            rows.append(base_row)
            continue

        if block == "ANTISACCADE":
            # cue_color на step=2 start
            cue = trial[(trial["step"]==2) & (trial["edge"]=="start")]["color"]
            if cue.empty:
                rows.append(base_row)
                continue

            onset_s = float(trial[(trial["step"]==2) & (trial["edge"]=="start")]["t_s"].min())
            cue_color = cue.iloc[0]

            base_row["onset_s"] = onset_s
            base_row["cue_color"] = cue_color

            rt = gaze_based_rt(rec, onset_s, cfg)
            base_row["qc_trial_valid"] = int(rt["qc_trial_valid"])
            base_row["rt_found"] = int(rt["rt_found"])
            base_row["rt_gaze_ms"] = rt["rt_gaze_ms"]
            base_row["response_dir"] = rt["response_dir"]

            # фиксации trial
            t0, t1 = trial_window_from_parsed(pe, trial_id, onset_s)
            base_row.update(fixation_trial_stats(rec, t0, t1))

            rows.append(base_row)
            continue

    return pd.DataFrame(rows)