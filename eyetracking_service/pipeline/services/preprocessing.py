import numpy as np

def add_time_seconds(df, ts_col, t0_ns, out_col):
    if df is None or df.empty:
        return df
    df = df.copy()
    df[out_col] = (df[ts_col].astype(np.int64) - int(t0_ns)) / 1e9
    return df

def normalize_time(rec: dict) -> dict:
    """
    normalize_time(rec) -> rec
    По ТЗ:
      t0_ns = info["start_time"]
      t_s = (timestamp_ns - t0_ns)/1e9
    """
    t0_ns = rec["info"].get("start_time")
    if t0_ns is None:
        # fallback: min timestamp из gaze/events
        candidates = []
        if not rec["gaze"].empty:
            candidates.append(rec["gaze"]["timestamp [ns]"].min())
        if not rec["events"].empty:
            candidates.append(rec["events"]["timestamp [ns]"].min())
        if not candidates:
            raise ValueError(f"Не удалось определить t0_ns для recording {rec['recording_id']}")
        t0_ns = int(min(candidates))

    rec["t0_ns"] = int(t0_ns)

    rec["gaze"] = add_time_seconds(rec["gaze"], "timestamp [ns]", t0_ns, "t_s")
    rec["events"] = add_time_seconds(rec["events"], "timestamp [ns]", t0_ns, "t_s")

    rec["fixations"] = add_time_seconds(rec["fixations"], "start timestamp [ns]", t0_ns, "t_start_s")
    rec["fixations"] = add_time_seconds(rec["fixations"], "end timestamp [ns]", t0_ns, "t_end_s")

    rec["saccades"] = add_time_seconds(rec["saccades"], "start timestamp [ns]", t0_ns, "t_start_s")
    rec["saccades"] = add_time_seconds(rec["saccades"], "end timestamp [ns]", t0_ns, "t_end_s")

    rec["blinks"] = add_time_seconds(rec["blinks"], "start timestamp [ns]", t0_ns, "t_start_s")
    rec["blinks"] = add_time_seconds(rec["blinks"], "end timestamp [ns]", t0_ns, "t_end_s")

    rec["imu"] = add_time_seconds(rec["imu"], "timestamp [ns]", t0_ns, "t_s")
    rec["world_timestamps"] = add_time_seconds(rec["world_timestamps"], "timestamp [ns]", t0_ns, "t_s")

    return rec