from __future__ import annotations

import tempfile
import logging
from pathlib import Path
import librosa
from antropy import sample_entropy, detrended_fluctuation
from scipy.spatial.distance import pdist, squareform
from scipy.stats import variation
import zipfile
import tempfile
from typing import Optional



import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

def extract_audio_from_video(video_path: str | Path,
                              output_audio_path: str | Path) -> tuple[str, int]:

    y, sr = librosa.load(str(video_path), sr=None)
    sf.write(str(output_audio_path), y, sr)
    logger.info('Аудио извлечено: %s → %s (sr=%d)', video_path, output_audio_path, sr)
    return str(output_audio_path), sr


def calculate_correlation_dimension(signal: np.ndarray,
                                      emb_dim: int = 10,
                                      tau: int = 1) -> float:
    
    epsilons = np.logspace(-3, 0, 50)
    n_points = len(signal) - (emb_dim - 1) * tau
    if n_points <= 0:
        return float('nan')

    embedded = np.zeros((n_points, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = signal[i * tau: i * tau + n_points]

    std = np.std(embedded)
    if std == 0:
        return float('nan')
    embedded = (embedded - np.mean(embedded, axis=0)) / std

    dist_matrix = squareform(pdist(embedded))

    C_epsilon = []
    for eps in epsilons:
        C = np.sum(dist_matrix < eps) - n_points
        C_epsilon.append(C / (n_points * (n_points - 1) + 1e-10))

    log_eps = np.log(epsilons)
    log_C   = np.log(np.array(C_epsilon))
    mask    = np.isfinite(log_eps) & np.isfinite(log_C)
    if np.sum(mask) < 2:
        return float('nan')

    coeffs = np.polyfit(log_eps[mask], log_C[mask], 1)
    D2 = float(coeffs[0])
    return D2 if 0 < D2 < emb_dim * 1.5 else float('nan')


def extract_audio_features(file_path, status_label):
    y, sr = librosa.load(file_path, sr=44100)
    # частотные характеристики
    f0 = librosa.yin(y, fmin=60, fmax=300)
    f0 = f0[f0 > 0] #чистка записей

    features = {
        'MDVP:Fo(Hz)': np.mean(f0),
        'MDVP:Fhi(Hz)': np.max(f0),
        'MDVP:Flo(Hz)': np.min(f0)
    }
    # параметры дрожания
    diffs = np.diff(f0)
    abs_diffs = np.abs(diffs)
    periods = 1 / f0[f0 > 0] #частота в периоды
    mean_period = np.mean(periods)

    features.update({
        'MDVP:Jitter(%)': variation(f0) * 100,
        'MDVP:Jitter(Abs)': np.mean(abs_diffs),
        'MDVP:RAP': np.mean(0.5 * abs_diffs[:-1] + 0.5 * abs_diffs[1:]),
        'MDVP:PPQ': np.mean(np.convolve(abs_diffs, [0.2]*5, 'valid')),
        'Jitter:DDP': np.mean(np.abs(diffs[:-1] - diffs[1:]))
    })
    # параметры дрожания амплитуды
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    amplitudes = [np.max(np.abs(frame)) for frame in frames.T]
    amp_diffs = np.abs(np.diff(amplitudes))

    features.update({
        'MDVP:Shimmer': np.std(amplitudes) / np.mean(amplitudes),
        'MDVP:Shimmer(dB)': 20 * np.log10(np.std(amplitudes) / np.mean(amplitudes)),
        'Shimmer:APQ3': np.mean(amp_diffs[:len(amp_diffs)-2] + amp_diffs[1:-1] + amp_diffs[2:]) / 3,
        'Shimmer:APQ5': np.mean(np.convolve(amp_diffs, [0.2]*5, 'valid')),
        'MDVP:APQ': np.mean(np.abs(periods - mean_period)) / mean_period * 100,
        'Shimmer:DDA': np.mean(np.abs(np.diff(amplitudes, 2)))
    })
    # параметры шума
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)

    features.update({
        'NHR': np.sum(percussive**2) / (np.sum(harmonic**2) + 1e-10),
        'HNR': 10 * np.log10(np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)),
        'status': status_label
    })
    # нелинейные параметры
    features.update({
        'RPDE': sample_entropy(f0, order=2, metric='chebyshev'),
        'DFA': detrended_fluctuation(f0),
        'spread1': np.quantile(f0, 0.9) - np.quantile(f0, 0.1),
        'spread2': np.std(f0) / np.mean(f0),
    })

    features['D2'] = calculate_correlation_dimension(f0)
    features.update({
        'PPE': sample_entropy(f0, order=2, metric='euclidean')
    })

    return features

def extract_recording_id_from_path(video_path: str | Path) -> str:
    return Path(video_path).parent.name.encode("utf-8", errors="ignore").decode("utf-8")


def process_audio_file(file_path: str | Path,
                        status_label: int | None = None) -> dict:

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if file_path.suffix.lower() == '.zip':
        temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            try:
                filename = info.filename.encode('cp437').decode('cp866')
            except Exception:
                filename = info.filename

            target_path = Path(temp_dir) / filename
            parent = target_path.parent

            if parent.exists() and parent.is_file():
                parent.unlink()

            if target_path.exists():
                if target_path.is_dir():
                    continue
                target_path.unlink()

            parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(info) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

        # ищем первый audio/video файл
        media_files = list(Path(temp_dir).rglob('*.mp4'))

        if not media_files:
            media_files = list(Path(temp_dir).rglob('*.wav'))

        if not media_files:
            raise ValueError('В zip нет audio/video файлов')

        # заменяем file_path на найденный файл
        file_path = media_files[0]

    # Видеофайл — нужно извлечь аудио
    is_video = suffix in {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

    if is_video:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            extract_audio_from_video(file_path, tmp_path)
            features = extract_audio_features(tmp_path, status_label)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        features = extract_audio_features(file_path, status_label)

    if not features:
        return {}

    mapping = {
        'MDVP:Fo(Hz)':      'fo_hz',
        'MDVP:Fhi(Hz)':     'fhi_hz',
        'MDVP:Flo(Hz)':     'flo_hz',
        'MDVP:Jitter(%)':   'jitter_pct',
        'MDVP:Jitter(Abs)': 'jitter_abs',
        'MDVP:RAP':         'rap',
        'MDVP:PPQ':         'ppq',
        'Jitter:DDP':       'jitter_ddp',
        'MDVP:Shimmer':     'shimmer',
        'MDVP:Shimmer(dB)': 'shimmer_db',
        'Shimmer:APQ3':     'shimmer_apq3',
        'Shimmer:APQ5':     'shimmer_apq5',
        'MDVP:APQ':         'mdvp_apq',
        'Shimmer:DDA':      'shimmer_dda',
        'NHR':              'nhr',
        'HNR':              'hnr',
        'RPDE':             'rpde',
        'DFA':              'dfa',
        'spread1':          'spread1',
        'spread2':          'spread2',
        'PPE':              'ppe',
        'status':           'status',
    }

    db_fields = {mapping[k]: v for k, v in features.items() if k in mapping}

    for key, val in db_fields.items():
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            db_fields[key] = None

    db_fields['recording_id'] = extract_recording_id_from_path(file_path)

    if status_label is not None:
        db_fields['label_text'] = 'patient' if status_label == 1 else 'healthy'

    return db_fields