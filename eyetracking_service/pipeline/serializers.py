from rest_framework import serializers
from .models import Recording, SubjectQC, TrialMetrics, BlockSummary, SpeechRecord, MultimodalPrediction

# class ZipUploadSerializer(serializers.Serializer):
#     zip_file = serializers.FileField(help_text="ZIP архив с данными")

class ZipUploadSerializer(serializers.Serializer):
    zip_file = serializers.FileField(help_text='ZIP архив с данными')
    label = serializers.IntegerField(required=False, allow_null=True, help_text='Известный диагноз: 0=healthy, 1=patient. Опционально.')

    def validate_label(self, value):
        if value is not None and value not in (0, 1):
            raise serializers.ValidationError('label должен быть 0 или 1')
        return value


class AudioZipUploadSerializer(serializers.Serializer):
    zip_file = serializers.FileField(help_text='ZIP-архив с аудио/видео файлами пациента')
    label = serializers.IntegerField(required=False, allow_null=True,help_text='Известный диагноз: 0=healthy, 1=patient. Опционально.')

    def validate_label(self, value):
        if value is not None and value not in (0, 1):
            raise serializers.ValidationError('label должен быть 0 или 1')
        return value

class RecordingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recording
        fields = "__all__"


class SubjectQCSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectQC
        fields = "__all__"


class TrialMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrialMetrics
        fields = "__all__"


class BlockSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = BlockSummary
        fields = "__all__"


class SpeechRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeechRecord
        fields = '__all__'
        read_only_fields = ('created_at',)


class MultimodalPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MultimodalPrediction
        fields = '__all__'
        read_only_fields = ('created_at',)

class UnimodalPredictSerializer(serializers.Serializer):
    """
    Параметры унимодального предсказания.
    Используется когда есть данные только по одной модальности.
    """
    modality = serializers.ChoiceField(
        choices=[
        ('eye', 'eye'),
        ('speech', 'speech'),
    ],
        help_text='Какую модальность использовать: "eye" или "speech"'
    )
    threshold = serializers.FloatField(
        required=False, default=0.5,
        min_value=0.0, max_value=1.0,
        help_text='Порог классификации P(patient).'
    )