from django.urls import path
from .views import *


urlpatterns = [
    path("upload/single/", UploadSingleRecordingZip.as_view()),
    # TrialMetrics
    path("trials/", TrialMetricsListView.as_view()),
    path("trials/<str:recording_id>/", TrialMetricsByRecording.as_view()),
    path("trials/block/<str:block>/", TrialMetricsByBlock.as_view()),
    # BlockSummary
    path("blocks/", BlockSummaryListView.as_view()),
    path("blocks/<str:recording_id>/", BlockSummaryByRecording.as_view()),
    # QC
    path("qc/", SubjectQCListView.as_view()),
    path("qc/<str:recording_id>/", SubjectQCByRecording.as_view()),
    # Upload
    path("upload/multiple/", UploadMultipleZips.as_view()),
    # Speech
    path('speech/upload/', UploadAudioView.as_view(), name='speech-upload'),
    path('speech/', SpeechRecordListView.as_view(), name='speech-list'),
    path('speech/<str:recording_id>/', SpeechRecordByRecording.as_view(),name='speech-by-recording'),
    #ML
    path('ml/predict/<str:recording_id>/', MultimodalPredictView.as_view(),name='ml-predict'),
    path('ml/history/<str:recording_id>/', MultimodalPredictionListView.as_view(),name='ml-history'),
    path('ml/predict-unimodal/<str:recording_id>/',UnimodalPredictView.as_view(), name='ml-predict-unimodal'),
    # PDF Отчёт
    path('report/pdf/<str:recording_id>/', PatientReportPDFView.as_view(),name='patient-report-pdf'),
]