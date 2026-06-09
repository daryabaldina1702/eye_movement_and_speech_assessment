from django.test import TestCase
from rest_framework.test import APIClient, APITestCase
from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
import io
import pandas as pd
import tempfile
import os

from pipeline.models import (
    Recording,
    SubjectQC,
    SpeechRecord,
    MultimodalPrediction,
    BlockSummary,
    TrialMetrics
)

#Вспомогательные функции
def make_recording(recording_id='test-rec-001', label=1, label_text='patient'):
    """Создаёт Recording в БД и возвращает объект."""
    return Recording.objects.create(
        recording_id=recording_id,
        zip_name='test.zip',
        label=label,
        label_text=label_text,
    )
 
def make_subject_qc(recording):
    """Создаёт SubjectQC с тестовыми значениями."""
    return SubjectQC.objects.create(
        recording=recording,
        duration_s=720.0,
        gaze_rate_hz=248.0,
        worn_ratio=0.97,
        n_gaze_samples=178000,
        blink_rate_per_min=18.0,
        n_fixations=900,
        fix_dur_ms_median=640.0,
        fix_dur_ms_mean=660.0,
        n_saccades=890,
        sacc_amp_deg_mean=8.3,
        sacc_peak_px_s_mean=None,
        imu_gyro_rms=0.04,
        qc_flag_low_worn=0,
        qc_flag_low_rate=0,
        qc_flag_high_motion=0,
    )
def make_speech_record(recording):
    """Создаёт SpeechRecord с тестовыми акустическими признаками."""
    return SpeechRecord.objects.create(
        recording=recording,
        fo_hz=120.0, fhi_hz=180.0, flo_hz=80.0,
        jitter_pct=0.005, jitter_abs=0.00003,
        rap=0.003, ppq=0.003, jitter_ddp=0.009,
        shimmer=0.04, shimmer_db=0.35,
        shimmer_apq3=0.02, shimmer_apq5=0.025,
        mdvp_apq=0.03, shimmer_dda=0.06,
        nhr=0.014, hnr=21.0,
        rpde=0.48, dfa=0.72,
        spread1=-5.5, spread2=0.22, ppe=0.20,
        status=1, label_text='patient',
    )
def make_prediction(recording):
    """Создаёт MultimodalPrediction с тестовыми значениями."""
    return MultimodalPrediction.objects.create(
        recording=recording,
        p_eye=0.72, p_speech=0.65,
        w_eye=0.5, w_speech=0.5,
        p_fused=0.685, threshold=0.5,
        prediction=1, label_text='patient', confidence=0.685,
        eye_model_name='RandomForest', speech_model_name='SVM',
    )
 
 
def _make_fake_zip() -> bytes:
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('recording_folder/placeholder.txt', 'test')
    return buf.getvalue()

_FAKE_QC_DF = pd.DataFrame([{
    'recording_id':      'test-rec-001',
    'zip_name':          'test.zip',
    'label':             1,
    'label_text':        'patient',
    'duration_s':        720.0,
    'gaze_rate_hz':      248.0,
    'worn_ratio':        0.97,
    'n_gaze_samples':    178000,
    'blink_rate_per_min':18.0,
    'n_fixations':       900,
    'fix_dur_ms_median': 640.0,
    'fix_dur_ms_mean':   660.0,
    'n_saccades':        890,
    'sacc_amp_deg_mean': 8.3,
    'sacc_peak_px_s_mean': None,
    'imu_gyro_rms':      0.04,
    'qc_flag_low_worn':  0,
    'qc_flag_low_rate':  0,
    'qc_flag_high_motion': 0,
    'recording_folder':  '/tmp/test',
}])
_FAKE_TRIAL_DF = pd.DataFrame()
_FAKE_BLOCK_DF = pd.DataFrame()
 
_FAKE_PREDICT_RESULT = {
    'p_eye':      0.72,
    'p_speech':   0.65,
    'w_eye':      0.5,
    'w_speech':   0.5,
    'p_fused':    0.685,
    'threshold':  0.5,
    'prediction': 1,
    'label_text': 'patient',
    'confidence': 0.685,
    'eye_model_name':    'RandomForest',
    'modality_used':'multimodal',
    'speech_model_name': 'SVM',
}
 
_FAKE_AUDIO_FIELDS = {
    'recording_id':  'audio-rec-001',
    'fo_hz':   120.0, 'fhi_hz': 180.0, 'flo_hz': 80.0,
    'jitter_pct': 0.005, 'jitter_abs': 0.00003,
    'rap': 0.003, 'ppq': 0.003, 'jitter_ddp': 0.009,
    'shimmer': 0.04, 'shimmer_db': 0.35,
    'shimmer_apq3': 0.02, 'shimmer_apq5': 0.025,
    'mdvp_apq': 0.03, 'shimmer_dda': 0.06,
    'nhr': 0.014, 'hnr': 21.0,
    'rpde': 0.48, 'dfa': 0.72,
    'spread1': -5.5, 'spread2': 0.22, 'ppe': 0.20,
    'status': 1, 'label_text': 'patient'
}
_FAKE_UNIMODAL_EYE = {**_FAKE_PREDICT_RESULT, 'modality': 'eye',
                       'p_speech': None, 'w_eye': 1.0, 'w_speech': 0.0}
_FAKE_UNIMODAL_SP  = {**_FAKE_PREDICT_RESULT, 'modality': 'speech',
                       'p_eye': None, 'w_eye': 0.0, 'w_speech': 1.0}
 

class BaseAPITest(TestCase):

    def setUp(self):
        self.client = APIClient()

        self.recording = Recording.objects.create(
            recording_id="test_patient",
            zip_name="test.zip",
            label=1,
            label_text="patient"
        )

    def test_recording_created(self):
        self.assertEqual(
            Recording.objects.count(),
            1
        )

class MultimodalPredictionAPITest(APITestCase):

    def setUp(self):

        self.recording = Recording.objects.create(
            recording_id="test_patient",
            zip_name="test.zip",
            label=0,
            label_text="healty"
        )

        self.qc = SubjectQC.objects.create(
            recording=self.recording,
            duration_s=100,
            gaze_rate_hz=30,
            worn_ratio=0.95,
            n_gaze_samples=1000,
            blink_rate_per_min=15,
            n_fixations=120,
            fix_dur_ms_median=200,
            fix_dur_ms_mean=220,
            n_saccades=90,
            sacc_amp_deg_mean=3.2,
            sacc_peak_px_s_mean=500,
            imu_gyro_rms=0.12,
            qc_flag_low_worn=0,
            qc_flag_low_rate=0,
            qc_flag_high_motion=0,
        )

        self.speech = SpeechRecord.objects.create(
            recording=self.recording,
            fo_hz=120,
            fhi_hz=150,
            flo_hz=90,
            jitter_pct=0.01,
            jitter_abs=0.0001,
            rap=0.003,
            ppq=0.004,
            jitter_ddp=0.009,
            shimmer=0.02,
            shimmer_db=0.3,
            shimmer_apq3=0.01,
            shimmer_apq5=0.02,
            mdvp_apq=0.03,
            shimmer_dda=0.04,
            nhr=0.01,
            hnr=25,
            rpde=0.45,
            dfa=0.70,
            spread1=-5.0,
            spread2=0.2,
            ppe=0.15,
        )

    @patch("pipeline.views.predict_multimodal")
    def test_multimodal_prediction(self, mock_predict):

        mock_predict.return_value = {
            "prediction": 0,
            "label_text": "healthy",
            "confidence": 0.91,
            "p_eye": 0.82,
            "p_speech": 0.95,
            "p_fused": 0.89,
            "w_eye": 0.5,
            "w_speech": 0.5,
            "threshold": 0.51,
            "eye_model_name": "svm_eye.pkl",
            "speech_model_name": "speech_eye.pkl",
        }

        url = reverse(
            "ml-predict",
            kwargs={
                "recording_id": self.recording.recording_id
            }
        )

        response = self.client.post(url)

        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK
        )

        self.assertEqual(
            response.data["prediction"],
            0
        )

        self.assertEqual(
            response.data["label_text"],
            "healthy"
        )

        mock_predict.assert_called_once()

class SpeechRecordReadTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.rec    = make_recording()
        self.speech = make_speech_record(self.rec)
 
    def test_speech_list_returns_all(self):
        resp = self.client.get('/speech/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
 
    def test_speech_by_recording_returns_correct(self):
        resp = self.client.get(f'/speech/{self.rec.recording_id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        self.assertAlmostEqual(resp.data[0]['fo_hz'], 120.0)

class UploadSingleZipTests(TestCase): 
    def setUp(self):
        self.client = APIClient()
        self.url = '/upload/single/'
 
    @patch('pipeline.views.run_pipeline',
           return_value=(_FAKE_QC_DF, _FAKE_TRIAL_DF, _FAKE_BLOCK_DF))
    def test_upload_creates_recording(self, mock_pipeline):
        """Успешная загрузка ZIP создаёт Recording в БД."""
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip())},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Recording.objects.count(), 1)
        self.assertEqual(Recording.objects.first().recording_id, 'test-rec-001')
 
    @patch('pipeline.views.run_pipeline',
           return_value=(_FAKE_QC_DF, _FAKE_TRIAL_DF, _FAKE_BLOCK_DF))
    def test_upload_with_label_sets_label_text(self, mock_pipeline):
        """label=0 при загрузке → label_text='healthy'."""
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip()), 'label': 0},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 200)
        rec = Recording.objects.first()
        self.assertEqual(rec.label, 0)
        self.assertEqual(rec.label_text, 'healthy')
 
    @patch('pipeline.views.run_pipeline',
           return_value=(_FAKE_QC_DF, _FAKE_TRIAL_DF, _FAKE_BLOCK_DF))
    def test_upload_creates_subject_qc(self, mock_pipeline):
        self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip())},
            format='multipart',
        )
        self.assertEqual(SubjectQC.objects.count(), 1)
 
    @patch('pipeline.views.run_pipeline',
           return_value=(_FAKE_QC_DF, _FAKE_TRIAL_DF, _FAKE_BLOCK_DF))
    def test_upload_idempotent_get_or_create(self, mock_pipeline):
        for _ in range(2):
            self.client.post(
                self.url,
                {'zip_file': io.BytesIO(_make_fake_zip())},
                format='multipart',
            )
        self.assertEqual(Recording.objects.count(), 1)
 
    def test_upload_without_file_returns_400(self):
        """Запрос без zip_file возвращает 400."""
        resp = self.client.post(self.url, {}, format='multipart')
        self.assertEqual(resp.status_code, 400)
 
    def test_upload_invalid_label_returns_400(self):
        """label=99 (недопустимое значение) возвращает 400."""
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip()), 'label': 99},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)
 
    def test_upload_response_contains_recording_id(self):
        with patch('pipeline.views.run_pipeline',
                   return_value=(_FAKE_QC_DF, _FAKE_TRIAL_DF, _FAKE_BLOCK_DF)):
            resp = self.client.post(
                self.url,
                {'zip_file': io.BytesIO(_make_fake_zip())},
                format='multipart',
            )
        self.assertIn('recording_id', resp.data)
        self.assertEqual(resp.data['recording_id'], 'test-rec-001')
 
class UploadAudioViewTests(TestCase):
    """Тесты загрузки ZIP с аудио/видео для извлечения речевых признаков."""
 
    def setUp(self):
        self.client = APIClient()
        self.url = '/speech/upload/'
 
    @patch('pipeline.views.process_audio_file', return_value=_FAKE_AUDIO_FIELDS.copy())
    def test_upload_audio_creates_speech_record(self, mock_proc):
        """Успешная загрузка создаёт SpeechRecord и Recording."""
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip()), 'label': 1},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(SpeechRecord.objects.count(), 1)
        self.assertEqual(Recording.objects.count(), 1)
 
    @patch('pipeline.views.process_audio_file', return_value=_FAKE_AUDIO_FIELDS.copy())
    def test_upload_audio_links_speech_to_recording(self, mock_proc):
        """SpeechRecord связывается с Recording через FK."""
        self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip()), 'label': 1},
            format='multipart',
        )
        sp = SpeechRecord.objects.first()
        rec = Recording.objects.first()
        self.assertEqual(sp.recording_id, rec.id)
 
    @patch('pipeline.views.process_audio_file', return_value={})
    def test_upload_empty_features_returns_400(self, mock_proc):
        """Если признаки не извлечены (пустой dict) → 400."""
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip())},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)
 
    def test_upload_audio_without_file_returns_400(self):
        """Запрос без файла → 400."""
        resp = self.client.post(self.url, {}, format='multipart')
        self.assertEqual(resp.status_code, 400)
 
    @patch('pipeline.views.process_audio_file', return_value=_FAKE_AUDIO_FIELDS.copy())
    def test_upload_audio_invalid_label_returns_400(self, mock_proc):
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip()), 'label': 5},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 400)
 
    @patch('pipeline.views.process_audio_file', return_value=_FAKE_AUDIO_FIELDS.copy())
    def test_upload_audio_response_has_fo_hz(self, mock_proc):
        resp = self.client.post(
            self.url,
            {'zip_file': io.BytesIO(_make_fake_zip())},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201)
        self.assertIn('fo_hz', resp.data)
 
class UnimodalPredictViewTests(TestCase): 
    def setUp(self):
        self.client = APIClient()
        self.rec = make_recording()
 
    def _url(self, recording_id=None, modality='eye', threshold=0.5):
        rid = recording_id or self.rec.recording_id
        return (f'/ml/predict-unimodal/{rid}/'
                f'?modality={modality}&threshold={threshold}')
 
    def test_unimodal_unknown_recording_returns_404(self):
        resp = self.client.post(self._url('nonexistent'))
        self.assertEqual(resp.status_code, 404)
 
    def test_unimodal_invalid_modality_returns_400(self):
        resp = self.client.post(self._url(modality='brain'))
        self.assertEqual(resp.status_code, 400)
 
    def test_unimodal_eye_no_qc_returns_422(self):
        resp = self.client.post(self._url(modality='eye'))
        self.assertEqual(resp.status_code, 422)
 
    def test_unimodal_speech_no_speech_record_returns_422(self):
        resp = self.client.post(self._url(modality='speech'))
        self.assertEqual(resp.status_code, 422)
 
    @patch('pipeline.views.predict_unimodal', return_value=_FAKE_UNIMODAL_EYE.copy())
    def test_unimodal_eye_success(self, mock_pred):
        make_subject_qc(self.rec)
        resp = self.client.post(self._url(modality='eye'))
        self.assertEqual(resp.status_code, 200)
        mock_pred.assert_called_once()
        args = mock_pred.call_args.kwargs
        self.assertEqual(args.get('modality'), 'eye')
 
    @patch('pipeline.views.predict_unimodal', return_value=_FAKE_UNIMODAL_SP.copy())
    def test_unimodal_speech_success(self, mock_pred):
        make_speech_record(self.rec)
        resp = self.client.post(self._url(modality='speech'))
        self.assertEqual(resp.status_code, 200)
        args = mock_pred.call_args.kwargs
        self.assertEqual(args.get('modality'), 'speech')
 
    @patch('pipeline.views.predict_unimodal', return_value=_FAKE_UNIMODAL_EYE.copy())
    def test_unimodal_saves_prediction(self, mock_pred):
        """Результат сохраняется в MultimodalPrediction."""
        make_subject_qc(self.rec)
        self.client.post(self._url(modality='eye'))
        self.assertEqual(MultimodalPrediction.objects.count(), 1)
 
    @patch('pipeline.views.predict_unimodal', return_value=_FAKE_UNIMODAL_EYE.copy())
    def test_unimodal_custom_threshold_passed(self, mock_pred):
        """Кастомный порог threshold=0.7 передаётся в predict_unimodal."""
        make_subject_qc(self.rec)
        self.client.post(self._url(modality='eye', threshold=0.7))
        args = mock_pred.call_args.kwargs
        self.assertAlmostEqual(args.get('threshold'), 0.7)

class PatientReportPDFViewTests(TestCase):
    """Тесты генерации PDF-отчёта."""

    def setUp(self):
        self.client = APIClient()
        self.rec = make_recording()
        self.url = f'/report/pdf/{self.rec.recording_id}/'

    # helper: создаём реально доступный файл
    def _make_pdf(self):
        f = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        f.write(b'%PDF-1.4 fake pdf content')
        f.flush()
        f.close()
        return f.name

    def _safe_remove(self, path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError:
            # Windows иногда держит handle — просто пропускаем
            pass

    # ── 200 OK ─────────────────────────────────────────
    @patch('pipeline.views.generate_patient_report')
    def test_pdf_returns_200_and_pdf_content_type(self, mock_gen):
        make_prediction(self.rec)
        make_subject_qc(self.rec)
        make_speech_record(self.rec)

        tmp_path = self._make_pdf()
        mock_gen.return_value = tmp_path

        self.addCleanup(self._safe_remove, tmp_path)

        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'application/pdf')

    # ── QC/Speech отсутствуют ─────────────────────────
    @patch('pipeline.views.generate_patient_report')
    def test_pdf_works_without_qc_and_speech(self, mock_gen):
        make_prediction(self.rec)

        tmp_path = self._make_pdf()
        mock_gen.return_value = tmp_path

        self.addCleanup(self._safe_remove, tmp_path)

        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 200)

        kwargs = mock_gen.call_args.kwargs
        self.assertIsNone(kwargs['qc_data'])
        self.assertIsNone(kwargs['speech_data'])

    # ── filename ──────────────────────────────────────
    @patch('pipeline.views.generate_patient_report')
    def test_pdf_filename_contains_recording_id(self, mock_gen):
        make_prediction(self.rec)

        tmp_path = self._make_pdf()
        mock_gen.return_value = tmp_path

        self.addCleanup(self._safe_remove, tmp_path)

        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 200)

        disposition = resp.get('Content-Disposition', '')
        self.assertIn(self.rec.recording_id.replace(' ', '_'), disposition)

    # ── prediction data ───────────────────────────────
    @patch('pipeline.views.generate_patient_report')
    def test_pdf_passes_correct_prediction_data(self, mock_gen):
        pred = make_prediction(self.rec)

        tmp_path = self._make_pdf()
        mock_gen.return_value = tmp_path

        self.addCleanup(self._safe_remove, tmp_path)

        self.client.get(self.url)

        self.assertTrue(mock_gen.called)

        call_kwargs = mock_gen.call_args.kwargs

        self.assertEqual(call_kwargs['recording_id'], self.rec.recording_id)

        pr = call_kwargs['prediction_result']
        self.assertAlmostEqual(pr['p_fused'], pred.p_fused, places=4)
        self.assertEqual(pr['prediction'], pred.prediction)
        self.assertEqual(pr['label_text'], pred.label_text)
        self.assertAlmostEqual(pr['confidence'], pred.confidence, places=4)
        self.assertEqual(pr['eye_model_name'], pred.eye_model_name)

    # ── error 500 ─────────────────────────────────────
    @patch('pipeline.views.generate_patient_report',
           side_effect=Exception('ReportLab error'))
    def test_pdf_generation_error_returns_500(self, mock_gen):
        make_prediction(self.rec)

        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 500)
        self.assertIn('Ошибка генерации PDF', resp.data['error'])

    
    @patch('pipeline.views.generate_patient_report')
    def test_pdf_has_content_length(self, mock_gen):
        make_prediction(self.rec)

        tmp_path = self._make_pdf()
        mock_gen.return_value = tmp_path

        self.addCleanup(self._safe_remove, tmp_path)

        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 200)
        self.assertIn('Content-Length', resp)
        self.assertGreater(int(resp['Content-Length']), 0)
 

class SubjectQCTests(TestCase):
    """Тесты GET-эндпоинтов QC-метрик."""
    def setUp(self):
        self.client = APIClient()
        self.rec = make_recording()
        self.qc  = make_subject_qc(self.rec)
 
    def test_qc_list_returns_all(self):
        resp = self.client.get('/qc/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
 
    def test_qc_by_recording_returns_correct(self):
        resp = self.client.get(f'/qc/{self.rec.recording_id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        self.assertEqual(resp.data[0]['gaze_rate_hz'], 248.0)
 
    def test_qc_by_recording_nonexistent_returns_empty(self):
        resp = self.client.get('/qc/does-not-exist/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, [])

class BlockSummaryTests(TestCase):
    """Тесты GET-эндпоинтов Block Summary."""
 
    def setUp(self):
        self.client = APIClient()
        self.rec = make_recording()
        BlockSummary.objects.create(
            recording=self.rec, block='GAP', n_trials=20,
            rt_gaze_ms_mean=220.0, rt_gaze_ms_median=215.0,
            rt_gaze_ms_std=30.0, direction_error_rate=0.05,
        )
 
    def test_blocks_list(self):
        resp = self.client.get('/blocks/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
 
    def test_blocks_by_recording(self):
        resp = self.client.get(f'/blocks/{self.rec.recording_id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data[0]['block'], 'GAP')
class TrialMetricsTests(TestCase):
    """Тесты GET-эндпоинтов Trial Metrics."""
 
    def setUp(self):
        self.client = APIClient()
        self.rec    = make_recording()
        TrialMetrics.objects.create(
            recording=self.rec, block='PREDICTION', trial_id=1
        )
 
    def test_trials_list(self):
        resp = self.client.get('/trials/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
 
    def test_trials_by_recording(self):
        resp = self.client.get(f'/trials/{self.rec.recording_id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data[0]['block'], 'PREDICTION')
 