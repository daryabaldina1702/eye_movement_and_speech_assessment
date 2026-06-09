from pathlib import Path
import zipfile
import pandas as pd
from .io_utils import unzip_cached


# =========================
# LABEL DETECTION
# =========================
def infer_label_from_zipname(zip_path: Path) -> tuple[int | None, str]:
    name = zip_path.stem.lower()

    healthy_tokens = ["hc", "healthy", "control", "ctl", "norm", "normal"]
    patient_tokens = ["pat", "patient", "ad", "alz", "alzheimer", "mci", "dement"]

    if any(tok in name for tok in patient_tokens):
        return 1, "patient"
    if any(tok in name for tok in healthy_tokens):
        return 0, "healthy"
    return None, "unknown"


# =========================
# ZIP LISTING
# =========================
def list_zip_files(dataset_dir: Path) -> list[Path]:
    zips = sorted(dataset_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"В папке {dataset_dir} не найдено .zip файлов")
    return zips


# =========================
# FIND RECORDINGS
# =========================
def find_recording_folders(root: Path) -> list[Path]:
    root = Path(root)
    recs = []

    for p in root.rglob("info.json"):
        folder = p.parent
        if (folder / "gaze.csv").exists():
            recs.append(folder.resolve())

    return sorted(list({r for r in recs}))


# =========================
# BUILD ZIP INDEX
# =========================
def build_zip_index(dataset_dir: Path) -> pd.DataFrame:
    zip_files = list_zip_files(dataset_dir)

    zip_index = []
    for zp in zip_files:
        label, label_text = infer_label_from_zipname(zp)

        zip_index.append({
            "zip_path": str(zp),
            "zip_name": zp.name,
            "zip_stem": zp.stem,
            "label": label,
            "label_text": label_text,
        })

    zip_index_df = pd.DataFrame(zip_index)

    # manual override (если нужно)
    MANUAL_LABELS = {}

    if MANUAL_LABELS:
        for k, v in MANUAL_LABELS.items():
            mask = zip_index_df["zip_name"] == k
            zip_index_df.loc[mask, "label"] = v
            zip_index_df.loc[mask, "label_text"] = "patient" if v == 1 else "healthy"

    return zip_index_df


# =========================
# BUILD DATASET INDEX
# =========================
def build_dataset_index(dataset_dir: Path, work_dir: Path) -> pd.DataFrame:
    """
    Главная функция:
    zip → unzip → recordings → dataset_index
    """

    zip_index_df = build_zip_index(dataset_dir)

    records = []

    for _, row in zip_index_df.iterrows():
        zp = Path(row["zip_path"])

        extracted_root = unzip_cached(zp, work_dir)
        rec_folders = find_recording_folders(extracted_root)

        for rf in rec_folders:
            records.append({
                "zip_name": row["zip_name"],
                "zip_stem": row["zip_stem"],
                "label": row["label"],
                "label_text": row["label_text"],
                "recording_folder": str(rf),
                "recording_id": rf.name,
            })

    dataset_index = pd.DataFrame(records)

    print("Всего recordings:", len(dataset_index))

    return dataset_index