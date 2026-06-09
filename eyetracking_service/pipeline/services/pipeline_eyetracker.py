import pandas as pd
from .config import CFG
from pathlib import Path
from .preprocessing import normalize_time
from .io_utils import load_recording
from .parsing import infer_block_windows, parse_trial_events
from .qc import compute_qc
from .trial import compute_trial_metrics_for_block
from .block import summarize_blocks
from .dataset_service import build_dataset_index

def process_one_recording(recording_folder: Path, cfg=CFG) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      qc_row (dict)
      trial_metrics_df
      block_summary_df
    """
    rec = normalize_time(load_recording(recording_folder))

    bw = infer_block_windows(rec["events"])
    pe = parse_trial_events(rec["events"], bw)

    qc_row = compute_qc(rec, cfg)

    # trial metrics по блокам
    tm_list = []
    for b in ["PREDICTION","GAP","OVERLAP","DECISION","ANTISACCADE"]:
        tm = compute_trial_metrics_for_block(rec, pe, b, cfg)
        if not tm.empty:
            tm_list.append(tm)

    trial_metrics = pd.concat(tm_list, ignore_index=True) if tm_list else pd.DataFrame()
    block_summary = summarize_blocks(trial_metrics) if not trial_metrics.empty else pd.DataFrame()

    return qc_row, trial_metrics, block_summary


def run_pipeline(dataset_dir: Path, work_dir: Path):
    all_qc = []
    all_trials = []
    all_blocks = []

    dataset_index = build_dataset_index(dataset_dir, work_dir)
    print(dataset_index)

    for _, r in dataset_index.iterrows():
        folder = Path(r["recording_folder"])

        qc_row, tm, bs = process_one_recording(
            folder,
            CFG
        )

        # метаданные
        qc_row.update({
            "zip_name": r["zip_name"],
            "zip_stem": r["zip_stem"],
            "label": r["label"],
            "label_text": r["label_text"],
            "recording_folder": str(folder),
        })

        all_qc.append(qc_row)

        if not tm.empty:
            tm = tm.copy()
            tm["zip_name"] = r["zip_name"]
            tm["zip_stem"] = r["zip_stem"]
            tm["label"] = r["label"]
            tm["label_text"] = r["label_text"]
            all_trials.append(tm)

        if not bs.empty:
            bs = bs.copy()
            bs["zip_name"] = r["zip_name"]
            bs["zip_stem"] = r["zip_stem"]
            bs["label"] = r["label"]
            bs["label_text"] = r["label_text"]
            all_blocks.append(bs)

    subject_qc_df = pd.DataFrame(all_qc)
    trial_metrics_df = (
        pd.concat(all_trials, ignore_index=True)
        if all_trials else pd.DataFrame()
    )
    block_summary_df = (
        pd.concat(all_blocks, ignore_index=True)
        if all_blocks else pd.DataFrame()
    )

    return subject_qc_df, trial_metrics_df, block_summary_df

# all_qc = []
# all_trials = []
# all_blocks = []
# dataset_index = build_dataset_index(dataset_dir, work_dir)

# for _, r in dataset_index.iterrows():
#     folder = Path(r["recording_folder"])
#     qc_row, tm, bs = process_one_recording(folder, CFG)

#     # добавим метки healthy/patient на уровне recording
#     qc_row["zip_name"] = r["zip_name"]
#     qc_row["zip_stem"] = r["zip_stem"]
#     qc_row["label"] = r["label"]
#     qc_row["label_text"] = r["label_text"]
#     qc_row["recording_folder"] = str(folder)

#     all_qc.append(qc_row)

#     if not tm.empty:
#         tm = tm.copy()
#         tm["zip_name"] = r["zip_name"]
#         tm["zip_stem"] = r["zip_stem"]
#         tm["label"] = r["label"]
#         tm["label_text"] = r["label_text"]
#         all_trials.append(tm)

#     if not bs.empty:
#         bs = bs.copy()
#         bs["zip_name"] = r["zip_name"]
#         bs["zip_stem"] = r["zip_stem"]
#         bs["label"] = r["label"]
#         bs["label_text"] = r["label_text"]
#         all_blocks.append(bs)

#     subject_qc_df = pd.DataFrame(all_qc)
#     trial_metrics_df = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
#     block_summary_df = pd.concat(all_blocks, ignore_index=True) if all_blocks else pd.DataFrame()
#     return subject_qc_df, trial_metrics_df, block_summary_df