import pandas as pd
import re

def infer_block_windows(events: pd.DataFrame) -> pd.DataFrame:
    """
    infer_block_windows(events) -> df(block, part, t_start, t_end)
    По ТЗ:
      intro: START_<B>_BLOCK_START .. START_<B>_BLOCK_END
      trials: START_<B>_BLOCK_END .. END_<B>_BLOCK_START
      outro: END_<B>_BLOCK_START .. END_<B>_BLOCK_END
    Для CALIB: CALIB_BLOCK_START..CALIB_BLOCK_END
    """
    ev = events.sort_values("timestamp [ns]").copy()
    rows = []

    # CALIB
    if (ev["name"] == "CALIB_BLOCK_START").any() and (ev["name"] == "CALIB_BLOCK_END").any():
        t_start = ev.loc[ev["name"]=="CALIB_BLOCK_START", "t_s"].min()
        t_end   = ev.loc[ev["name"]=="CALIB_BLOCK_END", "t_s"].min()
        rows.append({"block":"CALIB", "part":"trials", "t_start":t_start, "t_end":t_end})

    for b in ["PREDICTION","GAP","OVERLAP","DECISION","ANTISACCADE"]:
        s1 = f"START_{b}_BLOCK_START"
        s2 = f"START_{b}_BLOCK_END"
        e1 = f"END_{b}_BLOCK_START"
        e2 = f"END_{b}_BLOCK_END"
        if not ((ev["name"]==s1).any() and (ev["name"]==s2).any() and (ev["name"]==e1).any() and (ev["name"]==e2).any()):
            continue

        intro_start = ev.loc[ev["name"]==s1, "t_s"].min()
        intro_end   = ev.loc[ev["name"]==s2, "t_s"].min()
        trials_start = intro_end
        trials_end   = ev.loc[ev["name"]==e1, "t_s"].min()
        outro_start  = trials_end
        outro_end    = ev.loc[ev["name"]==e2, "t_s"].min()

        rows += [
            {"block":b, "part":"intro",  "t_start":intro_start, "t_end":intro_end},
            {"block":b, "part":"trials", "t_start":trials_start, "t_end":trials_end},
            {"block":b, "part":"outro",  "t_start":outro_start, "t_end":outro_end},
        ]

    return pd.DataFrame(rows).sort_values(["t_start","block","part"]).reset_index(drop=True)


def parse_event_name(name):
    m = re.match(r"(\d+)\.(\d+)_(.*)_(start|end)", str(name))
    if not m:
        return None

    return {
        "trial_id": int(m.group(1)),
        "step": int(m.group(2)),
        "tag": m.group(3),
        "edge": m.group(4)
    }

def parse_trial_events(events: pd.DataFrame, block_windows: pd.DataFrame) -> pd.DataFrame:
    """
    parse_trial_events(events, block_windows) -> df
    Выдаёт parsed events таблицу по ТЗ:
      block, trial_id, step, tag, edge, side, color, correctness, param_num, t_s
    Парсим только внутри part='trials' окон.
    """
    out = []
    for _, bw in block_windows[block_windows["part"]=="trials"].iterrows():
        b = bw["block"]
        seg = events[(events["t_s"]>=bw["t_start"]) & (events["t_s"]<=bw["t_end"])].copy()
        seg = seg.sort_values("timestamp [ns]")

        for _, r in seg.iterrows():
            p = parse_event_name(r["name"])
            if p is None:
                continue
            p["block"] = b
            p["t_s"] = float(r["t_s"])
            out.append(p)

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    cols = ["block","trial_id","step","tag","edge","side","color","correctness","param_num","t_s"]
    df = df.reindex(columns=cols)
    return df.sort_values(["block","trial_id","step","t_s"]).reset_index(drop=True)