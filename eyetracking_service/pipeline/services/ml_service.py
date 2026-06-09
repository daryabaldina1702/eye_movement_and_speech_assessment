from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
import joblib
from django.conf import settings

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EYE_MODEL_FILE    = 'svm_eye.pkl'
DEFAULT_SPEECH_MODEL_FILE = 'svm_speech.pkl'

EYE_FEATURE_NAMES = [
    'duration_s', 'gaze_rate_hz', 'worn_ratio', 'n_gaze_samples',
    'blink_rate_per_min', 'n_fixations', 'fix_dur_ms_median',
    'fix_dur_ms_mean', 'n_saccades', 'sacc_amp_deg_mean',
    'sacc_peak_px_s_mean', 'imu_gyro_rms',
]

SPEECH_FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'PPE',
]

SPEECH_DB_TO_FEAT = {
    'fo_hz':       'MDVP:Fo(Hz)',
    'fhi_hz':      'MDVP:Fhi(Hz)',
    'flo_hz':      'MDVP:Flo(Hz)',
    'jitter_pct':  'MDVP:Jitter(%)',
    'jitter_abs':  'MDVP:Jitter(Abs)',
    'rap':         'MDVP:RAP',
    'ppq':         'MDVP:PPQ',
    'jitter_ddp':  'Jitter:DDP',
    'shimmer':     'MDVP:Shimmer',
    'shimmer_db':  'MDVP:Shimmer(dB)',
    'shimmer_apq3':'Shimmer:APQ3',
    'shimmer_apq5':'Shimmer:APQ5',
    'mdvp_apq':    'MDVP:APQ',
    'shimmer_dda': 'Shimmer:DDA',
    'nhr':         'NHR',
    'hnr':         'HNR',
    'rpde':        'RPDE',
    'dfa':         'DFA',
    'spread1':     'spread1',
    'spread2':     'spread2',
    'ppe':         'PPE',
}

def _get_models_dir() -> Path:
    return Path(getattr(settings, 'ML_MODELS_DIR', 'ml'))


def load_model(filename: str):
    path = _get_models_dir() / filename
    print(path)
    if not path.exists():
        logger.warning('Модель не найдена: %s', path)
        return None
    try:
        model = joblib.load(path)
        logger.info('Модель загружена: %s', path)
        return model
    except Exception as exc:
        logger.error('Ошибка загрузки %s: %s', path, exc)
        return None


def _dict_to_vector(feature_dict: dict, feature_names: list) -> np.ndarray:
    row = [feature_dict.get(f, np.nan) for f in feature_names]
    arr = np.array(row, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.reshape(1, -1)


def _predict_proba(model, X: np.ndarray) -> float:
    if model is None:
        return 0.5  # нет модели → неопределённость

    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0, 1])

    # SVM / SVR без probability=True
    dv = model.decision_function(X)[0]
    return float(1.0 / (1.0 + np.exp(-dv)))  # sigmoid


def predict_multimodal(
    eye_features: Optional[dict],
    speech_features: Optional[dict],
    w_eye: float = 0.5,
    threshold: float = 0.5,
    eye_model_file:    str = DEFAULT_EYE_MODEL_FILE,
    speech_model_file: str = DEFAULT_SPEECH_MODEL_FILE,
) -> dict:
    """
    Мультимодальное предсказание через Soft Voting.

    Если одна модальность отсутствует (None) — автоматически
    переключается в унимодальный режим (вес = 1.0).

    Параметры
    ----------
    eye_features     : dict признаков Eye Tracking (EYE_FEATURE_NAMES), или None
    speech_features  : dict признаков Speech (SPEECH_FEATURE_NAMES), или None
    w_eye            : вес Eye Tracking; Speech-вес = 1 - w_eye
    threshold        : порог P(patient) для классификации
    eye_model_file   : имя .pkl файла Eye-модели
    speech_model_file: имя .pkl файла Speech-модели
    """
    w_speech = 1.0 - w_eye

    eye_model    = load_model(eye_model_file)
    speech_model = load_model(speech_model_file)

    eye_model_name    = type(eye_model).__name__    if eye_model    else 'NoModel'
    speech_model_name = type(speech_model).__name__ if speech_model else 'NoModel'

    has_eye    = eye_features    is not None
    has_speech = speech_features is not None

    # ── Eye вероятность ───────────────────────────────────────────
    if has_eye and eye_model is not None:
        p_eye = _predict_proba(eye_model, _dict_to_vector(eye_features, EYE_FEATURE_NAMES))
    else:
        p_eye = None
        if not has_eye:
            logger.info('Eye-данные отсутствуют')

    # ── Speech вероятность ────────────────────────────────────────
    if has_speech and speech_model is not None:
        p_speech = _predict_proba(speech_model, _dict_to_vector(speech_features, SPEECH_FEATURE_NAMES))
    else:
        p_speech = None
        if not has_speech:
            logger.info('Speech-данные отсутствуют')

    # ── Определяем режим и финальную вероятность ─────────────────
    if p_eye is not None and p_speech is not None:
        p_fused      = w_eye * p_eye + w_speech * p_speech
        modality_used = 'multimodal'
    elif p_eye is not None:
        p_fused      = p_eye
        w_eye        = 1.0
        w_speech     = 0.0
        modality_used = 'eye_only'
        logger.info('Автопереключение в eye_only (нет Speech данных/модели)')
    elif p_speech is not None:
        p_fused      = p_speech
        w_eye        = 0.0
        w_speech     = 1.0
        modality_used = 'speech_only'
        logger.info('Автопереключение в speech_only (нет Eye данных/модели)')
    else:
        # Нет ни данных, ни моделей → неопределённость
        p_fused      = 0.5
        modality_used = 'none'
        logger.warning('Нет доступных данных и моделей — возвращается p=0.5')

    return _build_result(
        p_fused, threshold, p_eye, p_speech,
        w_eye, w_speech,
        eye_model_name, speech_model_name,
        modality_used,
    )


def _build_result(p_fused: float, threshold: float,
                  p_eye: Optional[float], p_speech: Optional[float],
                  w_eye: float, w_speech: float,
                  eye_model_name: str, speech_model_name: str,
                  modality_used: str) -> dict:
    prediction = 1 if p_fused >= threshold else 0
    label_text = 'patient' if prediction == 1 else 'healthy'
    confidence = p_fused if prediction == 1 else (1.0 - p_fused)
    return {
        'p_eye':            round(p_eye, 4)    if p_eye    is not None else None,
        'p_speech':         round(p_speech, 4) if p_speech is not None else None,
        'w_eye':            round(w_eye, 4),
        'w_speech':         round(w_speech, 4),
        'p_fused':          round(p_fused, 4),
        'threshold':        round(threshold, 4),
        'prediction':       prediction,
        'label_text':       label_text,
        'confidence':       round(confidence, 4),
        'modality_used':    modality_used,
        'eye_model_name':   eye_model_name,
        'speech_model_name':speech_model_name,
    }

def predict_unimodal(
    modality: str,
    features: dict,
    threshold: float = 0.5,
    model_file: str = '',
) -> dict:
    """
    Унимодальное предсказание — только Eye Tracking или только Speech.

    Параметры
    ----------
    modality   : 'eye' или 'speech'
    features   : словарь признаков соответствующей модальности
    threshold  : порог P(patient)
    model_file : имя .pkl файла; если пустой — выбирается по умолчанию
    """
    if modality not in ('eye', 'speech'):
        raise ValueError(f'modality должен быть "eye" или "speech", получено: {modality!r}')

    if modality == 'eye':
        mfile          = model_file or DEFAULT_EYE_MODEL_FILE
        feature_names  = EYE_FEATURE_NAMES
        model          = load_model(mfile)
        model_name     = type(model).__name__ if model else 'NoModel'
        p_val          = _predict_proba(model, _dict_to_vector(features, feature_names))
        return _build_result(
            p_val, threshold,
            p_eye=p_val, p_speech=None,
            w_eye=1.0, w_speech=0.0,
            eye_model_name=model_name,
            speech_model_name='NoModel',
            modality_used='eye_only',
        )
    else:  # speech
        mfile          = model_file or DEFAULT_SPEECH_MODEL_FILE
        feature_names  = SPEECH_FEATURE_NAMES
        model          = load_model(mfile)
        model_name     = type(model).__name__ if model else 'NoModel'
        p_val          = _predict_proba(model, _dict_to_vector(features, feature_names))
        return _build_result(
            p_val, threshold,
            p_eye=None, p_speech=p_val,
            w_eye=0.0, w_speech=1.0,
            eye_model_name='NoModel',
            speech_model_name=model_name,
            modality_used='speech_only',
        )

def speech_record_to_feature_dict(speech_record) -> dict:
    db_dict = {
        field: getattr(speech_record, field)
        for field in SPEECH_DB_TO_FEAT
    }
    return {SPEECH_DB_TO_FEAT[k]: v for k, v in db_dict.items()}


def subjectqc_to_feature_dict(qc_record) -> dict:
    return {
        'duration_s':          qc_record.duration_s,
        'gaze_rate_hz':        qc_record.gaze_rate_hz,
        'worn_ratio':          qc_record.worn_ratio,
        'n_gaze_samples':      qc_record.n_gaze_samples,
        'blink_rate_per_min':  qc_record.blink_rate_per_min,
        'n_fixations':         qc_record.n_fixations,
        'fix_dur_ms_median':   qc_record.fix_dur_ms_median,
        'fix_dur_ms_mean':     qc_record.fix_dur_ms_mean,
        'n_saccades':          qc_record.n_saccades,
        'sacc_amp_deg_mean':   qc_record.sacc_amp_deg_mean,
        'sacc_peak_px_s_mean': qc_record.sacc_peak_px_s_mean,
        'imu_gyro_rms':        qc_record.imu_gyro_rms,
    }
