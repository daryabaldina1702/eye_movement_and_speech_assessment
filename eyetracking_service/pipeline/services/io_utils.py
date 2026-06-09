import zipfile
import pandas as pd
import json
from pathlib import Path

def infer_label_from_zipname(zip_path: Path) -> tuple[int | None, str]:
    """
    Возвращает:
      label: 0=healthy/control, 1=patient, None=unknown
      label_text: 'healthy'/'patient'/'unknown'
    Рекомендуемая схема имен:
      HC_001.zip, CONTROL_*.zip -> healthy
      PAT_001.zip, AD_*.zip, MCI_*.zip -> patient
    """
    name = zip_path.stem.lower()

    healthy_tokens = ["hc", "healthy", "control", "ctl", "norm", "normal"]
    patient_tokens = ["pat", "patient", "ad", "alz", "alzheimer", "mci", "dement"]

    if any(tok in name for tok in patient_tokens):
        return 1, "patient"
    if any(tok in name for tok in healthy_tokens):
        return 0, "healthy"
    return None, "unknown"

def list_zip_files(dataset_dir: Path):
    zips = sorted(dataset_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"В папке {dataset_dir} не найдено .zip файлов")
    return zips

# def unzip_cached(zip_path: Path, work_dir: Path) -> Path:
#     out_dir = work_dir / zip_path.stem
#     if out_dir.exists():
#         return out_dir

#     out_dir.mkdir(parents=True, exist_ok=True)
#     with zipfile.ZipFile(zip_path, "r") as z:
#         z.extractall(out_dir)
#     return out_dir
def unzip_cached(zip_path: Path, work_dir: Path) -> Path:
    out_dir = work_dir / zip_path.stem

    if out_dir.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():

            try:
                member.filename = member.filename.encode('cp437').decode('cp866')
            except Exception:
                pass

            z.extract(member, out_dir)

    return out_dir

def find_recording_folders(root: Path):
    recs = []
    for p in root.rglob("info.json"):
        folder = p.parent
        if (folder / "gaze.csv").exists():
            recs.append(folder.resolve())
    return list(set(recs))

def read_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_safe(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_recording(folder: Path) -> dict:
    folder = Path(folder)
    info = read_json(folder / "info.json")
    scene_camera = read_json(folder / "scene_camera.json")

    rec = {
        "folder": folder,
        "recording_id": folder.name,
        "info": info,
        "scene_camera": scene_camera,
        "gaze": read_csv_safe(folder / "gaze.csv"),
        "fixations": read_csv_safe(folder / "fixations.csv"),
        "saccades": read_csv_safe(folder / "saccades.csv"),
        "blinks": read_csv_safe(folder / "blinks.csv"),
        "events": read_csv_safe(folder / "events.csv"),
        "imu": read_csv_safe(folder / "imu.csv"),
        "world_timestamps": read_csv_safe(folder / "world_timestamps.csv"),
    }
    return rec
def clean_nan_values(d: dict):
    """
    Заменяет все NaN на None (для Django)
    """
    return {
        k: (None if pd.isna(v) else v)
        for k, v in d.items()
    }