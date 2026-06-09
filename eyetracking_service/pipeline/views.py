import tempfile, zipfile
import os
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .serializers import *
from .services.pipeline_eyetracker import run_pipeline
from .services.io_utils import clean_nan_values
from .models import Recording, SubjectQC, TrialMetrics, BlockSummary
import pandas as pd
from django.http import FileResponse, HttpResponse
import numpy as np
from .services.ml_service import (
    predict_multimodal, speech_record_to_feature_dict,
    subjectqc_to_feature_dict, predict_unimodal
)
from .services.speech_service import process_audio_file
from .services.report_service import generate_patient_report

def _label_to_text(label) -> str:
    if label == 0:
        return 'healthy'
    if label == 1:
        return 'patient'
    return ''

class UploadSingleRecordingZip(APIView):
    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        request_body=ZipUploadSerializer,
        responses={
            201: openapi.Response('Создана новая запись', RecordingSerializer),
            200: openapi.Response('Запись уже существует, обновлена', RecordingSerializer),
            400: 'Ошибка валидации или обработки',
        },
        operation_description=('Загрузка ZIP-архива')
    )
    def post(self, request):
        serializer = ZipUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        zip_file     = request.FILES['zip_file']
        label_input  = serializer.validated_data.get('label', None)
        label_text   = _label_to_text(label_input) if label_input is not None else ''

        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir    = Path(tmp_dir)
            dataset_dir = work_dir / 'dataset'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            zip_path = dataset_dir / zip_file.name
            with open(zip_path, 'wb') as f:
                for chunk in zip_file.chunks():
                    f.write(chunk)

            subject_qc_df, trial_df, block_df = run_pipeline(
                dataset_dir=dataset_dir, work_dir=work_dir
            )

            qc_row = subject_qc_df.iloc[0].to_dict()
            zip_name = qc_row["zip_name"]

            tm = trial_df if not trial_df.empty else pd.DataFrame()
            bs = block_df if not block_df.empty else pd.DataFrame()

            rec_obj, _ = Recording.objects.get_or_create(
                recording_id=qc_row["recording_id"],
                defaults={
                    "zip_name": zip_name,
                    "label": label_input,
                    "label_text": label_text,
                    }
                )
            SubjectQC.objects.create(
                recording=rec_obj,
                **{k: v for k, v in qc_row.items()
                   if k in [f.name for f in SubjectQC._meta.fields]}
            )

            for _, r in tm.iterrows():
                data = clean_nan_values(r.to_dict())
                data.pop("recording_id", None)
                data.pop("recording", None)

                TrialMetrics.objects.create(
                    recording=rec_obj,
                    **data
                )

            for _, r in bs.iterrows():
                data = clean_nan_values(r.to_dict())
                data.pop("recording_id", None)
                data.pop("recording", None)

                BlockSummary.objects.create(
                    recording=rec_obj,
                    **data
                )

        return Response({
            "status": "ok",
            "recording_id": rec_obj.recording_id
        })



class TrialMetricsListView(APIView):
    @swagger_auto_schema(operation_description="Все trial metrics")
    def get(self, request):
        qs = TrialMetrics.objects.all()
        return Response(TrialMetricsSerializer(qs, many=True).data)

class TrialMetricsByRecording(APIView):
    @swagger_auto_schema(operation_description="Trial metrics по recording_id")
    def get(self, request, recording_id):
        qs = TrialMetrics.objects.filter(recording__recording_id=recording_id)
        return Response(TrialMetricsSerializer(qs, many=True).data)
    
class TrialMetricsByBlock(APIView):
    @swagger_auto_schema(operation_description="Trial metrics по block")
    def get(self, request, block):
        qs = TrialMetrics.objects.filter(block=block)
        return Response(TrialMetricsSerializer(qs, many=True).data)

class SubjectQCListView(APIView):
    @swagger_auto_schema(operation_description="Все QC")
    def get(self, request):
        qs = SubjectQC.objects.all()
        return Response(SubjectQCSerializer(qs, many=True).data)

class SubjectQCByRecording(APIView):
    @swagger_auto_schema(operation_description="QC по recording_id")
    def get(self, request, recording_id):
        qs = SubjectQC.objects.filter(recording__recording_id=recording_id)
        return Response(SubjectQCSerializer(qs, many=True).data)

class BlockSummaryListView(APIView):
    @swagger_auto_schema(operation_description="Все block summary")
    def get(self, request):
        qs = BlockSummary.objects.all()
        return Response(BlockSummarySerializer(qs, many=True).data)

class BlockSummaryByRecording(APIView):
    @swagger_auto_schema(operation_description="Block summary по recording_id")
    def get(self, request, recording_id):
        qs = BlockSummary.objects.filter(recording__recording_id=recording_id)
        return Response(BlockSummarySerializer(qs, many=True).data)

class UploadMultipleZips(APIView):
    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        request_body=ZipUploadSerializer,
        operation_description="Загрузка архива с множеством zip"
    )
    def post(self, request):

        zip_file = request.FILES.get("zip_file")

        if not zip_file:
            return Response({"error": "no file"}, status=400)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            outer_zip_path = tmp_dir / zip_file.name

            # сохраняем внешний архив
            with open(outer_zip_path, "wb") as f:
                for chunk in zip_file.chunks():
                    f.write(chunk)

            # распаковка outer.zip
            extract_dir = tmp_dir / "dataset"
            extract_dir.mkdir()

            with zipfile.ZipFile(outer_zip_path, 'r') as z:
                z.extractall(extract_dir)

            subject_qc_df, trial_df, block_df = run_pipeline(
                dataset_dir=extract_dir,
                work_dir=tmp_dir
            )

            created_recordings = []
            for _, qc_row in subject_qc_df.iterrows():

                qc_data = clean_nan_values(qc_row.to_dict())

                rec_obj = Recording.objects.create(
                    recording_id=qc_data["recording_id"],
                    zip_name=qc_data["zip_name"],
                    label=qc_data["label"],
                    label_text=qc_data["label_text"]
                )

                SubjectQC.objects.create(
                    recording=rec_obj,
                    **{k: v for k, v in qc_data.items()
                       if k in [f.name for f in SubjectQC._meta.fields]}
                )

                created_recordings.append(rec_obj)
            for _, r in trial_df.iterrows():

                data = clean_nan_values(r.to_dict())

                rec = Recording.objects.get(
                    recording_id=data["recording_id"]
                )

                TrialMetrics.objects.create(
                    recording=rec,
                    **data
                )

            for _, r in block_df.iterrows():

                data = clean_nan_values(r.to_dict())

                rec = Recording.objects.get(
                    recording_id=data["recording_id"]
                )

                BlockSummary.objects.create(
                    recording=rec,
                    **data
                )

        return Response({
            "status": "ok",
            "recordings_created": [r.recording_id for r in created_recordings]
        })

class UploadAudioView(APIView):

    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        request_body=ZipUploadSerializer,
        responses={
            201: SpeechRecordSerializer,
            400: 'Ошибка обработки архива',
        },
        operation_description=(
            'Загрузка ZIP-архива с audio/video файлами пациента'
        )
    )
    def post(self, request):

        serializer = AudioZipUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        zip_file         = request.FILES['zip_file']
        label_input      = serializer.validated_data.get('label', None)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.zip'
        ) as tmp:

            for chunk in zip_file.chunks():
                tmp.write(chunk)

            tmp_zip_path = tmp.name

            try:
                db_fields = process_audio_file(
                    file_path=tmp_zip_path,
                    status_label=label_input
                )
            except Exception as exc:
                return Response({'error': f'Ошибка извлечения признаков: {exc}'}, status=500)

            if not db_fields:
                return Response(
                    {'error': 'Не удалось извлечь признаки. '
                              'Убедитесь, что ZIP содержит аудио/видео со звуком.'},
                    status=400
                )

        label_text = _label_to_text(label_input) if label_input is not None else ''

        rec_obj, created = Recording.objects.get_or_create(
        recording_id=db_fields['recording_id'],
        defaults={
            'zip_name': zip_file.name,
            'label': label_input,
            'label_text': label_text,
        }
    )

        db_fields.pop("recording_id", None)
        speech_rec = SpeechRecord.objects.create(
            recording=rec_obj,
            **db_fields,
        )

        return Response(SpeechRecordSerializer(speech_rec).data, status=201)

class SpeechRecordListView(APIView):
    @swagger_auto_schema(operation_description='Все Speech Records')
    def get(self, request):
        return Response(
            SpeechRecordSerializer(SpeechRecord.objects.all(), many=True).data
        )
class SpeechRecordByRecording(APIView):

    def get(self, request, recording_id):
        qs = SpeechRecord.objects.filter(
            recording__recording_id=recording_id
        )

        return Response(
            SpeechRecordSerializer(qs, many=True).data
        )
    

class MultimodalPredictView(APIView):
    @swagger_auto_schema(
        responses={
            200: MultimodalPredictionSerializer,
            404: 'Recording не найден',
            422: 'Недостаточно данных для предсказания',
        },
        operation_description=(
            'Классификация пациента'
        )
    )
    def post(self, request, recording_id):
        try:
            rec = Recording.objects.get(
                recording_id=recording_id
            )

        except Recording.DoesNotExist:

            return Response(
                {
                    'error': (
                        f'Recording "{recording_id}" не найден'
                    )
                },
                status=404
            )
        eye_features = None

        qc = SubjectQC.objects.filter(recording=rec).first()

        if qc:
            eye_features = subjectqc_to_feature_dict(qc)

        speech_features = None

        speech = SpeechRecord.objects.filter(recording=rec).first()

        if speech:
            speech_features = speech_record_to_feature_dict(speech)

        if eye_features is None and speech_features is None:

            return Response(
                {
                    'error': (
                        'Нет Eye Tracking или Speech данных '
                        'для данной записи'
                    )
                },
                status=422
            )

        result = predict_multimodal(
            eye_features=eye_features,
            speech_features=speech_features,
        )

        pred_obj = MultimodalPrediction.objects.create(
            recording=rec,

            p_eye=result['p_eye'],
            p_speech=result['p_speech'],

            w_eye=result['w_eye'],
            w_speech=result['w_speech'],

            p_fused=result['p_fused'],
            threshold=result['threshold'],

            prediction=result['prediction'],
            label_text=result['label_text'],
            confidence=result['confidence'],

            eye_model_name=result['eye_model_name'],
            speech_model_name=result['speech_model_name'],
        )

        return Response(
            MultimodalPredictionSerializer(pred_obj).data,
            status=200
        )

class MultimodalPredictionListView(APIView):
    @swagger_auto_schema(
        operation_description='История предсказаний по recording_id'
    )
    def get(self, request, recording_id):
        try:
            rec = Recording.objects.get(recording_id=recording_id)
        except Recording.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        qs = MultimodalPrediction.objects.filter(recording=rec)
        return Response(MultimodalPredictionSerializer(qs, many=True).data)


class UnimodalPredictView(APIView):

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                name='modality',
                in_=openapi.IN_QUERY,
                description='Выбор модальности',
                type=openapi.TYPE_STRING,
                enum=['eye', 'speech'],
                required=True,
            ),
            openapi.Parameter(
                name='threshold',
                in_=openapi.IN_QUERY,
                description='Порог классификации',
                type=openapi.TYPE_NUMBER,
                default=0.5,
                required=False,
            ),
        ],
        responses={
            200: MultimodalPredictionSerializer,
            404: 'Recording не найден',
            422: 'Нет данных по запрошенной модальности',
        },
        operation_description=(
            'Унимодальное предсказание — только Eye Tracking или только Speech'
        )
    )
    def post(self, request, recording_id):

        modality = request.query_params.get('modality')
        threshold = float(request.query_params.get('threshold', 0.5))

        if modality not in ['eye', 'speech']:
            return Response(
                {'error': 'modality должна быть eye или speech'},
                status=400
            )

        try:
            rec = Recording.objects.get(recording_id=recording_id)
        except Recording.DoesNotExist:
            return Response(
                {'error': f'Recording "{recording_id}" не найден'},
                status=404
            )

        if modality == 'eye':
            qc = SubjectQC.objects.filter(recording=rec).first()
            if not qc:
                return Response(
                    {'error': f'Нет Eye Tracking данных для записи "{recording_id}"'},
                    status=422
                )
            features = subjectqc_to_feature_dict(qc)

        else:  
            speech = (
                SpeechRecord.objects.filter(recording=rec).first()
            )
            if not speech:
                return Response(
                    {'error': f'Нет Speech данных для записи "{recording_id}"'},
                    status=422
                )
            features = speech_record_to_feature_dict(speech)

        result = predict_unimodal(
            modality=modality,
            features=features,
            threshold=threshold
        )

        pred_obj = MultimodalPrediction.objects.create(
            recording=rec,
            p_eye=result['p_eye'],
            p_speech=result['p_speech'],
            w_eye=result['w_eye'],
            w_speech=result['w_speech'],
            p_fused=result['p_fused'],
            threshold=result['threshold'],
            prediction=result['prediction'],
            label_text=result['label_text'],
            confidence=result['confidence'],
            modality_used=result['modality_used'],
            eye_model_name=result['eye_model_name'],
            speech_model_name=result['speech_model_name'],
        )

        return Response(MultimodalPredictionSerializer(pred_obj).data, status=200)

class PatientReportPDFView(APIView):
    """Генерирует PDF-отчёт по пациенту на основе последней классификации."""
 
    @swagger_auto_schema(
        responses={
            200: openapi.Response('PDF файл', schema=openapi.Schema(type=openapi.TYPE_FILE)),
            404: 'Recording не найден или нет предсказания',
        },
        operation_description=(
            'Генерирует PDF-отчёт по recording_id. '
            'Требует наличия предсказания — сначала вызовите /ml/predict/{recording_id}/.'
        )
    )
    def get(self, request, recording_id):
        # Запись
        try:
            rec = Recording.objects.get(recording_id=recording_id)
        except Recording.DoesNotExist:
            return Response({'error': f'Recording "{recording_id}" не найден'}, status=404)
 
        # Предсказание — только из БД, не запускаем новое
        pred_obj = MultimodalPrediction.objects.filter(recording=rec).first()
        if pred_obj is None:
            return Response(
                {'error': 'Предсказание не найдено. '
                          f'Сначала вызовите POST /ml/predict/{recording_id}/'},
                status=404
            )
 
        qc_data = None
        speech_data = None
        block_data = None

        qc = SubjectQC.objects.filter(recording=rec).first()
        if qc:
            qc_data = qc.to_feature_dict()

        speech = SpeechRecord.objects.filter(recording=rec).first()
        if speech:
            speech_data = {
                f.name: getattr(speech, f.name)
                for f in SpeechRecord._meta.fields
                if isinstance(getattr(speech, f.name), (int, float, type(None)))
                and f.name not in ('id', 'status')
            }

        blocks_qs = BlockSummary.objects.filter(recording=rec)
        if blocks_qs.exists():
            block_data = list(blocks_qs.values())
 
        # Генерация PDF
        try:
            pdf_path = generate_patient_report(
            recording_id=recording_id,
            prediction_result={
                'p_eye': pred_obj.p_eye,
                'p_speech': pred_obj.p_speech,
                'w_eye': pred_obj.w_eye,
                'w_speech': pred_obj.w_speech,
                'p_fused': pred_obj.p_fused,
                'threshold': pred_obj.threshold,
                'prediction': pred_obj.prediction,
                'label_text': pred_obj.label_text,
                'confidence': pred_obj.confidence,
                'eye_model_name': pred_obj.eye_model_name,
                'speech_model_name': pred_obj.speech_model_name,
            },
            qc_data=qc_data,
            speech_data=speech_data,
            block_data=block_data
        )
        except Exception as e:
            return Response(
                {'error': f'Ошибка генерации PDF: {e}'},
                status=500
            )

        safe_name = recording_id.replace(' ', '_')

        return FileResponse(
            open(pdf_path, 'rb'),
            as_attachment=True,
            filename=f'report_{safe_name}.pdf',
            content_type='application/pdf'
)