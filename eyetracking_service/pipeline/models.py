# pipeline_app/models.py

from django.db import models

class Recording(models.Model):
    recording_id = models.CharField(max_length=255, unique=True)
    zip_name = models.CharField(max_length=255)
    label      = models.IntegerField(null=True, blank=True, help_text='0=healthy, 1=patient. Опционально.')
    label_text = models.CharField(max_length=50, blank=True, default='', help_text='healthy / patient / unknown')

    def __str__(self):
        return self.recording_id


class SubjectQC(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE)

    duration_s = models.FloatField(null=True)
    gaze_rate_hz = models.FloatField(null=True)
    worn_ratio = models.FloatField(null=True)
    n_gaze_samples = models.IntegerField(null=True)
    blink_rate_per_min = models.FloatField(null=True)

    n_fixations = models.IntegerField(null=True)
    fix_dur_ms_median = models.FloatField(null=True)
    fix_dur_ms_mean = models.FloatField(null=True)

    n_saccades = models.IntegerField(null=True)
    sacc_amp_deg_mean = models.FloatField(null=True)
    sacc_peak_px_s_mean = models.FloatField(null=True)

    imu_gyro_rms = models.FloatField(null=True)

    qc_flag_low_worn = models.IntegerField()
    qc_flag_low_rate = models.IntegerField()
    qc_flag_high_motion = models.IntegerField()

    # Поле recording_folder из CSV не добавлено, так как это служебный путь,
    # не относящийся к метрикам качества. При необходимости можно добавить:
    recording_folder = models.CharField(max_length=500, null=True, blank=True)
    
    def to_feature_dict(self) -> dict:
        return {
            'duration_s':          self.duration_s,
            'gaze_rate_hz':        self.gaze_rate_hz,
            'worn_ratio':          self.worn_ratio,
            'n_gaze_samples':      self.n_gaze_samples,
            'blink_rate_per_min':  self.blink_rate_per_min,
            'n_fixations':         self.n_fixations,
            'fix_dur_ms_median':   self.fix_dur_ms_median,
            'fix_dur_ms_mean':     self.fix_dur_ms_mean,
            'n_saccades':          self.n_saccades,
            'sacc_amp_deg_mean':   self.sacc_amp_deg_mean,
            'sacc_peak_px_s_mean': self.sacc_peak_px_s_mean,
            'imu_gyro_rms':        self.imu_gyro_rms,
        }


class TrialMetrics(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE)

    # Основные поля
    block = models.CharField(max_length=50)
    trial_id = models.IntegerField()
    onset_s = models.FloatField(null=True)
    target_side = models.CharField(max_length=10, null=True)

    # Качество попытки
    qc_trial_valid = models.IntegerField(null=True)          # 0/1

    # Временные параметры реакции
    rt_found = models.IntegerField(null=True)                # 0/1
    rt_gaze_ms = models.FloatField(null=True)
    rt_saccade_found = models.IntegerField(null=True)        # 0/1
    rt_saccade_ms = models.FloatField(null=True)

    # Направление ответа и ошибка
    response_dir = models.CharField(max_length=10, null=True)
    direction_error = models.FloatField(null=True)           # 0.0 / 1.0
    express_like = models.IntegerField(null=True)            # 0/1

    # Достижение цели
    target_reached = models.IntegerField(null=True)          # 0/1
    time_to_target_ms = models.FloatField(null=True)

    # Позиционные и точностные метрики
    post_response_dx_px = models.FloatField(null=True)
    accuracy_sector = models.FloatField(null=True)           # 0.0 / 1.0 или сектор

    # Длительности фиксаций на пробу
    fix_dur_ms_mean_trial = models.FloatField(null=True)
    fix_dur_ms_median_trial = models.FloatField(null=True)

    # Количество прыжков / шагов к цели
    n_large_jumps_to_target = models.IntegerField(null=True)
    n_steps_to_target = models.IntegerField(null=True)

    # Длительность всей пробы и доля времени до цели
    trial_duration_ms = models.FloatField(null=True)
    time_to_target_share = models.FloatField(null=True)

    # Параметры принятия решения (для блоков DECISION)
    decision_rt_gaze_ms = models.FloatField(null=True)
    choice_side = models.CharField(max_length=10, null=True)
    correct_side = models.CharField(max_length=10, null=True)
    decision_correct = models.FloatField(null=True)          # 0.0 / 1.0

    # Цвет сигнала (для антисаккад)
    cue_color = models.CharField(max_length=20, null=True)
    zip_name = models.CharField(max_length=255,null=True )
    label = models.IntegerField(null=True)
    label_text = models.CharField(default='None', max_length=50)
    zip_stem = models.CharField(max_length=255, null=True)

    def __str__(self):
        return f"{self.recording.recording_id} - {self.block} - {self.trial_id}"


class BlockSummary(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE)

    block = models.CharField(max_length=50)

    n_trials = models.IntegerField()
    rt_gaze_ms_mean = models.FloatField(null=True)
    rt_gaze_ms_median = models.FloatField(null=True)
    rt_gaze_ms_std = models.FloatField(null=True)            # добавлено из CSV

    direction_error_rate = models.FloatField(null=True)
    express_like_rate = models.FloatField(null=True)
    target_reached_rate = models.FloatField(null=True)

    time_to_target_ms_mean = models.FloatField(null=True)

    gap_overlap_delta_rt_ms = models.FloatField(null=True)
    zip_name = models.CharField(max_length=255,null=True )
    label = models.IntegerField(null=True)
    label_text = models.CharField(default='None', max_length=50)
    zip_stem = models.CharField(max_length=255, null=True)

    def __str__(self):
        return f"{self.recording.recording_id} - {self.block}"
    
class SpeechRecord(models.Model):
    recording = models.ForeignKey(
        Recording, on_delete=models.CASCADE,
        null=True, blank=True, related_name='speech_records'
    )

    # Частота основного тона
    fo_hz  = models.FloatField(null=True, verbose_name='MDVP:Fo(Hz)')
    fhi_hz = models.FloatField(null=True, verbose_name='MDVP:Fhi(Hz)')
    flo_hz = models.FloatField(null=True, verbose_name='MDVP:Flo(Hz)')

    # Джиттер
    jitter_pct = models.FloatField(null=True, verbose_name='MDVP:Jitter(%)')
    jitter_abs = models.FloatField(null=True, verbose_name='MDVP:Jitter(Abs)')
    rap        = models.FloatField(null=True, verbose_name='MDVP:RAP')
    ppq        = models.FloatField(null=True, verbose_name='MDVP:PPQ')
    jitter_ddp = models.FloatField(null=True, verbose_name='Jitter:DDP')

    # Шиммер
    shimmer      = models.FloatField(null=True, verbose_name='MDVP:Shimmer')
    shimmer_db   = models.FloatField(null=True, verbose_name='MDVP:Shimmer(dB)')
    shimmer_apq3 = models.FloatField(null=True, verbose_name='Shimmer:APQ3')
    shimmer_apq5 = models.FloatField(null=True, verbose_name='Shimmer:APQ5')
    mdvp_apq     = models.FloatField(null=True, verbose_name='MDVP:APQ')
    shimmer_dda  = models.FloatField(null=True, verbose_name='Shimmer:DDA')

    # Соотношение шум/гармоника
    nhr = models.FloatField(null=True, verbose_name='NHR')
    hnr = models.FloatField(null=True, verbose_name='HNR')

    # Нелинейные динамические признаки
    rpde    = models.FloatField(null=True, verbose_name='RPDE')
    dfa     = models.FloatField(null=True, verbose_name='DFA')
    spread1 = models.FloatField(null=True, verbose_name='spread1')
    spread2 = models.FloatField(null=True, verbose_name='spread2')
    ppe     = models.FloatField(null=True, verbose_name='PPE')

    status     = models.IntegerField(null=True, help_text='0=healthy, 1=patient')
    label_text = models.CharField(max_length=50, null=True, blank=True)



    def to_feature_dict(self):
        return {
            'MDVP:Fo(Hz)':      self.fo_hz,
            'MDVP:Fhi(Hz)':     self.fhi_hz,
            'MDVP:Flo(Hz)':     self.flo_hz,
            'MDVP:Jitter(%)':   self.jitter_pct,
            'MDVP:Jitter(Abs)': self.jitter_abs,
            'MDVP:RAP':         self.rap,
            'MDVP:PPQ':         self.ppq,
            'Jitter:DDP':       self.jitter_ddp,
            'MDVP:Shimmer':     self.shimmer,
            'MDVP:Shimmer(dB)': self.shimmer_db,
            'Shimmer:APQ3':     self.shimmer_apq3,
            'Shimmer:APQ5':     self.shimmer_apq5,
            'MDVP:APQ':         self.mdvp_apq,
            'Shimmer:DDA':      self.shimmer_dda,
            'NHR':              self.nhr,
            'HNR':              self.hnr,
            'RPDE':             self.rpde,
            'DFA':              self.dfa,
            'spread1':          self.spread1,
            'spread2':          self.spread2,
            'PPE':              self.ppe,
        }

class MultimodalPrediction(models.Model):
    """
    Результат мультимодального предсказания по пациенту.
    Хранит вероятности каждого классификатора и финальный вердикт.
    """
    recording  = models.ForeignKey(Recording, on_delete=models.CASCADE,
                                   related_name='predictions')
    created_at = models.DateTimeField(auto_now_add=True)

    p_eye    = models.FloatField(null=True, help_text='P(patient) от Eye-классификатора')
    p_speech = models.FloatField(null=True, help_text='P(patient) от Speech-классификатора')
    w_eye    = models.FloatField(default=0.5)
    w_speech = models.FloatField(default=0.5)
    p_fused  = models.FloatField(null=True)
    threshold = models.FloatField(default=0.5)

    prediction = models.IntegerField(null=True, help_text='0=healthy, 1=patient')
    label_text = models.CharField(max_length=50, null=True, blank=True)
    confidence = models.FloatField(null=True)
    modality_used = models.CharField(max_length=20,blank=True,null=True)

    eye_model_name    = models.CharField(max_length=100, default='SVM')
    speech_model_name = models.CharField(max_length=100, default='SVM')

    class Meta:
        verbose_name = 'Multimodal Prediction'
        ordering = ['-created_at']

    def __str__(self):
        return (f"Pred:{self.recording.recording_id} "
                f"→ {self.label_text} ({self.p_fused:.3f})")