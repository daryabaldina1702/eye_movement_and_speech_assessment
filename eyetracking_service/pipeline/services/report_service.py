from __future__ import annotations

import io
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)


from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
Paragraph, Spacer, Table, TableStyle,
)
_RL_OK = True
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

pdfmetrics.registerFont(
    TTFont("Arial", FONT_PATH)
)

pdfmetrics.registerFont(
    TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf")
)

C_HEALTHY = '#2196F3'   
C_PATIENT = '#E91E63'   
C_NEUTRAL = '#78909C'   
C_BG      = '#F5F5F5'  
C_ACCENT  = '#7B1FA2'  


def _hex(h: str):
    """Конвертирует HEX-строку в ReportLab Color."""
    h = h.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return colors.Color(r, g, b)


def _fig_to_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _make_gauge_chart(p_fused: float, threshold: float = 0.5) -> bytes:
    """Шкала-барометр итоговой вероятности."""
    fig, ax = plt.subplots(figsize=(5, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Фон шкалы
    ax.barh(0.5, 1.0, left=0, height=0.4,
            color=C_BG, edgecolor='#BDBDBD', linewidth=1.5)
    # Заполненная часть
    color = C_PATIENT if p_fused >= threshold else C_HEALTHY
    ax.barh(0.5, p_fused, left=0, height=0.4, color=color, alpha=0.85)
    # Метка порога
    ax.axvline(threshold, ymin=0.2, ymax=0.85, color='black',
               linewidth=2, linestyle='--')
    ax.text(threshold, 0.92, f'Порог\n{threshold:.2f}',
            ha='center', va='bottom', fontsize=7, color='black')
    # Значение
    ax.text(p_fused / 2, 0.5, f'{p_fused:.3f}',
            ha='center', va='center', fontsize=12,
            fontweight='bold', color='white')
    # Метки осей
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.text(v, 0.1, str(v), ha='center', va='top', fontsize=7, color='#616161')

    ax.set_title('P(patient) — финальная вероятность', fontsize=9, pad=2)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_probability_bars(p_eye: float, p_speech: float,
                            p_fused: float, threshold: float) -> bytes:
    """Горизонтальные бары трёх вероятностей."""
    fig, ax = plt.subplots(figsize=(6, 2.5))

    labels = ['Eye Tracking', 'Speech', 'Fused (финал)']
    values = [p_eye, p_speech, p_fused]
    bar_colors = [
        C_PATIENT if v >= threshold else C_HEALTHY
        for v in values
    ]

    bars = ax.barh(labels, values, color=bar_colors, alpha=0.85, height=0.5)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Порог {threshold:.2f}')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('P(patient)')
    ax.set_title('Вероятности классификаторов', fontsize=10)
    ax.legend(fontsize=8)

    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_eye_radar(qc_data: dict, ref_healthy: dict) -> bytes:
    """
    Радарная диаграмма — Eye Tracking пациента vs норма.
    qc_data / ref_healthy: dict {feature_name: value}
    """
    keys = [
        'gaze_rate_hz', 'worn_ratio', 'blink_rate_per_min',
        'fix_dur_ms_mean', 'sacc_amp_deg_mean', 'n_fixations',
    ]
    labels = [
        'Gaze Rate\n(Гц)', 'Worn\nRatio', 'Blink Rate\n(/мин)',
        'Fix Dur\n(мс)', 'Sacc Amp\n(°)', 'N Fix',
    ]

    # Нормализуем в [0,1] по max(patient, ref)
    def norm(feat, val):
        r = ref_healthy.get(feat, 1)
        p = qc_data.get(feat, 0) or 0
        m = max(abs(r), abs(p), 1e-6)
        return (val or 0) / m

    patient_vals = [norm(k, qc_data.get(k, 0)) for k in keys]
    ref_vals     = [norm(k, ref_healthy.get(k, 0)) for k in keys]

    N = len(keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    patient_vals += patient_vals[:1]
    ref_vals     += ref_vals[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, patient_vals, 'o-', color=C_PATIENT, lw=2, label='Пациент')
    ax.fill(angles, patient_vals, alpha=0.15, color=C_PATIENT)
    ax.plot(angles, ref_vals, 's--', color=C_HEALTHY, lw=2, label='Норма (среднее)')
    ax.fill(angles, ref_vals, alpha=0.10, color=C_HEALTHY)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('Eye Tracking: профиль пациента\nвс усреднённая норма', pad=15, fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_speech_bars(speech_data: dict) -> bytes:
    """Барчарт ключевых речевых признаков."""
    features = {
        'Fo (Гц)':       speech_data.get('fo_hz'),
        'HNR (дБ)':      speech_data.get('hnr'),
        'Jitter (%)':    speech_data.get('jitter_pct'),
        'Shimmer':       speech_data.get('shimmer'),
        'RPDE':          speech_data.get('rpde'),
        'DFA':           speech_data.get('dfa'),
        'PPE':           speech_data.get('ppe'),
        'NHR':           speech_data.get('nhr'),
    }
    # Нормы (приблизительные, из UCI Parkinsons Dataset)
    norms = {
        'Fo (Гц)':    154.2,
        'HNR (дБ)':   21.9,
        'Jitter (%)': 0.622,
        'Shimmer':    0.030,
        'RPDE':       0.498,
        'DFA':        0.718,
        'PPE':        0.207,
        'NHR':        0.025,
    }

    labels, vals, norm_vals, bar_colors = [], [], [], []
    for name, val in features.items():
        if val is None:
            continue
        labels.append(name)
        vals.append(val)
        norm_vals.append(norms.get(name, val))
        # Отклонение > 30% от нормы → подсветка
        norm = norms.get(name, val)
        bar_colors.append(
            C_PATIENT if abs(val - norm) / (abs(norm) + 1e-10) > 0.3 else C_HEALTHY
        )

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'Нет речевых данных', ha='center', va='center')
        ax.axis('off')
        return _fig_to_bytes(fig)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x - 0.2, vals, 0.35, color=bar_colors, alpha=0.85, label='Пациент')
    ax.bar(x + 0.2, norm_vals, 0.35, color=C_NEUTRAL, alpha=0.6, label='Норма')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_title('Речевые признаки: пациент vs норма', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_block_rt_chart(block_data: list[dict]) -> bytes:
    """Среднее RT по блокам айтрекинга."""
    if not block_data:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'Нет данных по блокам', ha='center', va='center')
        ax.axis('off')
        return _fig_to_bytes(fig)

    blocks = [d.get('block', '') for d in block_data]
    rts    = [d.get('rt_gaze_ms_mean') or 0 for d in block_data]
    errors = [d.get('direction_error_rate') or 0 for d in block_data]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # RT по блокам
    axes[0].bar(blocks, rts, color=C_ACCENT, alpha=0.8)
    axes[0].set_title('Среднее Reaction Time (мс) по блокам', fontsize=9)
    axes[0].set_ylabel('RT (мс)')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Ошибки по блокам
    axes[1].bar(blocks, errors, color=C_PATIENT, alpha=0.8)
    axes[1].set_title('Частота ошибок направления по блокам', fontsize=9)
    axes[1].set_ylabel('Direction Error Rate')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    return _fig_to_bytes(fig)


def generate_patient_report(
    recording_id: str,
    prediction_result: dict,
    qc_data: Optional[dict] = None,
    speech_data: Optional[dict] = None,
    block_data: Optional[list] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Генерирует PDF-отчёт по пациенту.

    Параметры
    ----------
    recording_id      : идентификатор пациента/записи
    prediction_result : словарь из ml_service.predict_multimodal()
    qc_data           : dict признаков Eye Tracking (SubjectQC.to_feature_dict())
    speech_data       : dict признаков Speech (поля SpeechRecord)
    block_data        : list[dict] данных BlockSummary
    output_path       : путь для сохранения PDF (если None — tempfile)

    Возвращает
    ----------
    str путь к сгенерированному PDF-файлу
    """

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix='.pdf',
            prefix=f'report_{recording_id}_'
        )
        output_path = tmp.name
        tmp.close()

    base_styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        'ReportTitle',
        parent=base_styles['Title'],
        fontSize=18, textColor=_hex('#1A237E'),
        spaceAfter=6,
    )
    style_h1 = ParagraphStyle(
        'H1', parent=base_styles['Heading1'],
        fontSize=13, textColor=_hex('#283593'),
        spaceBefore=14, spaceAfter=4,
        borderPad=2,
    )
    style_h2 = ParagraphStyle(
        'H2', parent=base_styles['Heading2'],
        fontSize=11, textColor=_hex('#4527A0'),
        spaceBefore=10, spaceAfter=3,
    )
    style_body = ParagraphStyle(
        'Body', parent=base_styles['Normal'],
        fontSize=9.5, leading=14, spaceAfter=4,
    )
    style_note = ParagraphStyle(
        'Note', parent=base_styles['Normal'],
        fontSize=8.5, textColor=_hex('#616161'),
        leading=12, spaceAfter=3,
    )
    style_verdict = ParagraphStyle(
        'Verdict', parent=base_styles['Normal'],
        fontSize=15, fontName='Arial-Bold',
        textColor=_hex(C_PATIENT if prediction_result.get('prediction') == 1 else C_HEALTHY),
        alignment=TA_CENTER, spaceAfter=6,
    )

    # ── Документ ─────────────────────────────────────────────────
    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id='main_frame'
    )
    template = PageTemplate(id='main', frames=frame)
    doc.addPageTemplates([template])

    story = []

    story.append(Paragraph(
        'Отчёт системы поддержки принятия решений', style_title
    ))
    story.append(Paragraph(
        f'Когнитивный скрининг | Мультимодальный анализ', style_body
    ))

    # Мета-таблица
    pr = prediction_result
    diag = pr.get('label_text', 'N/A').upper()
    conf = pr.get('confidence', 0)
    meta_data = [
        ['Параметр', 'Значение'],
        ['ID записи (Recording ID)', recording_id],
        ['Дата формирования отчёта', datetime.now().strftime('%d.%m.%Y %H:%M')],
        ['Eye-модель', pr.get('eye_model_name', 'N/A')],
        ['Speech-модель', pr.get('speech_model_name', 'N/A')],
        ['Версия системы', '2.0 (Multimodal)'],
    ]
    meta_table = Table(meta_data, colWidths=[6*cm, 11*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), _hex('#283593')),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Arial-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('BACKGROUND',   (0, 1), (-1, -1), _hex(C_BG)),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, _hex(C_BG)]),
        ('GRID',         (0, 0), (-1, -1), 0.5, _hex('#BDBDBD')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # ── 2. ИТОГОВЫЙ ВЕРДИКТ ───────────────────────────────────────
    story.append(Paragraph('2. Итоговый диагноз', style_h1))

    flag = 'PATIENT' if pr.get('prediction') == 1 else 'HEALTHY'
    story.append(Paragraph(
        f'{flag}   (уверенность: {conf:.1%})', style_verdict
    ))

    story.append(Paragraph(
        'Интерпретация: <b>PATIENT</b> — модель детектирует паттерны, '
        'характерные для когнитивных нарушений / болезни Паркинсона. '
        '<b>HEALTHY</b> — паттерны соответствуют норме.',
        style_note
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('3. Процесс принятия решения', style_h1))
    story.append(Paragraph(
        'Модель использует <b>Soft Voting (взвешенное усреднение вероятностей)</b>. '
        'Каждый базовый классификатор возвращает P(patient) — '
        'вероятность наличия патологии. Финальная вероятность вычисляется как '
        'взвешенная сумма: <i>P_fused = w_eye × P_eye + w_speech × P_speech</i>.',
        style_body
    ))

    prob_data = [
        ['Модальность', 'Модель', 'Вес', 'P(patient)', 'Вклад'],
        [
            'Eye Tracking',
            pr.get('eye_model_name', 'N/A'),
            f"{pr.get('w_eye', 0.5):.2f}",
            f"{pr.get('p_eye', 0):.4f}",
            f"{pr.get('w_eye', 0.5) * pr.get('p_eye', 0):.4f}",
        ],
        [
            'Speech',
            pr.get('speech_model_name', 'N/A'),
            f"{pr.get('w_speech', 0.5):.2f}",
            f"{pr.get('p_speech', 0):.4f}",
            f"{pr.get('w_speech', 0.5) * pr.get('p_speech', 0):.4f}",
        ],
        [
            'ФИНАЛ (Fused)',
            'Soft Voting',
            '1.00',
            f"{pr.get('p_fused', 0):.4f}",
            f"Порог: {pr.get('threshold', 0.5):.2f}",
        ],
    ]
    prob_table = Table(prob_data, colWidths=[4*cm, 3.5*cm, 2*cm, 2.5*cm, 2.5*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), _hex('#1A237E')),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Arial-Bold'),
        ('FONTNAME',     (0, 3), (-1, 3), 'Arial-Bold'),
        ('BACKGROUND',   (0, 3), (-1, 3), _hex('#E8EAF6')),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('GRID',         (0, 0), (-1, -1), 0.5, _hex('#BDBDBD')),
        ('ROWBACKGROUNDS',(0, 1), (-1, 2), [colors.white, _hex(C_BG)]),
        ('ALIGN',        (2, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*cm))

    prob_bytes = _make_probability_bars(
        pr.get('p_eye', 0.5), pr.get('p_speech', 0.5),
        pr.get('p_fused', 0.5), pr.get('threshold', 0.5)
    )
    story.append(Image(io.BytesIO(prob_bytes), width=14*cm, height=5.5*cm))
    story.append(Spacer(1, 0.2*cm))

    gauge_bytes = _make_gauge_chart(
        pr.get('p_fused', 0.5), pr.get('threshold', 0.5)
    )
    story.append(Image(io.BytesIO(gauge_bytes), width=12*cm, height=3.5*cm))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('4. Данные айтрекинга (Eye Tracking)', style_h1))

    if qc_data:
        story.append(Paragraph('4.1 QC-метрики пациента', style_h2))

        ref = {
            'duration_s':         720.0,
            'gaze_rate_hz':       250.0,
            'worn_ratio':         0.97,
            'n_gaze_samples':     180000,
            'blink_rate_per_min': 18.0,
            'n_fixations':        950,
            'fix_dur_ms_mean':    650.0,
            'n_saccades':         940,
            'sacc_amp_deg_mean':  8.5,
            'imu_gyro_rms':       0.05,
        }

        qc_table_data = [['Метрика', 'Пациент', 'Норма (ref)', 'Отклонение']]
        field_labels = {
            'duration_s':          'Длительность сессии (с)',
            'gaze_rate_hz':        'Частота записи взгляда (Гц)',
            'worn_ratio':          'Доля нахождения в оправе',
            'n_gaze_samples':      'Число сэмплов взгляда',
            'blink_rate_per_min':  'Частота морганий (/мин)',
            'n_fixations':         'Число фиксаций',
            'fix_dur_ms_mean':     'Ср. длит. фиксации (мс)',
            'n_saccades':          'Число саккад',
            'sacc_amp_deg_mean':   'Ср. амплитуда саккад (°)',
            'imu_gyro_rms':        'IMU-гироскоп RMS',
        }
        for key, label in field_labels.items():
            val = qc_data.get(key)
            ref_val = ref.get(key)
            if val is None:
                continue
            if ref_val and ref_val != 0:
                dev_pct = (val - ref_val) / abs(ref_val) * 100
                dev_str = f'{dev_pct:+.1f}%'
            else:
                dev_str = 'N/A'
            qc_table_data.append([
                label,
                f'{val:.2f}' if isinstance(val, float) else str(val),
                f'{ref_val:.2f}' if isinstance(ref_val, float) else str(ref_val or 'N/A'),
                dev_str,
            ])

        qc_table = Table(qc_table_data, colWidths=[6.5*cm, 3*cm, 3*cm, 2.5*cm])
        qc_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), _hex('#283593')),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, _hex(C_BG)]),
            ('GRID',          (0, 0), (-1, -1), 0.4, _hex('#BDBDBD')),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(qc_table)
        story.append(Spacer(1, 0.3*cm))

        # Радар
        story.append(Paragraph('4.2 Профиль пациента (нормализованный)', style_h2))
        radar_bytes = _make_eye_radar(qc_data, ref)
        story.append(Image(io.BytesIO(radar_bytes), width=10*cm, height=10*cm))
    else:
        story.append(Paragraph(
            'Данные Eye Tracking не загружены для данной записи.',
            style_note
        ))

    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('5. Речевые признаки (Speech)', style_h1))

    if speech_data:
        story.append(Paragraph('5.1 Ключевые параметры голоса', style_h2))

        sp_table_data = [['Признак', 'Значение', 'Описание']]
        sp_descriptions = {
            'fo_hz':       ('MDVP:Fo (Гц)', 'Средняя частота основного тона'),
            'fhi_hz':      ('MDVP:Fhi (Гц)', 'Максимальная частота'),
            'flo_hz':      ('MDVP:Flo (Гц)', 'Минимальная частота'),
            'jitter_pct':  ('Jitter (%)', 'Вариабельность периода, %'),
            'jitter_abs':  ('Jitter (Abs)', 'Абсолютный джиттер'),
            'shimmer':     ('Shimmer', 'Вариабельность амплитуды'),
            'shimmer_db':  ('Shimmer (dB)', 'Шиммер в дБ'),
            'nhr':         ('NHR', 'Шум / Гармоника'),
            'hnr':         ('HNR (дБ)', 'Гармоника / Шум'),
            'rpde':        ('RPDE', 'Нелинейная динамика'),
            'dfa':         ('DFA', 'Детрендированные флуктуации'),
            'spread1':     ('spread1', 'Диапазон частот (Q90-Q10)'),
            'spread2':     ('spread2', 'CV частоты основного тона'),
            'ppe':         ('PPE', 'Pitch Period Entropy'),
        }
        for field, (label, desc) in sp_descriptions.items():
            val = speech_data.get(field)
            if val is None:
                continue
            sp_table_data.append([
                label,
                f'{val:.4f}' if isinstance(val, float) else str(val),
                desc,
            ])

        sp_table = Table(sp_table_data, colWidths=[4*cm, 3.5*cm, 7.5*cm])
        sp_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), _hex('#4527A0')),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, _hex(C_BG)]),
            ('GRID',          (0, 0), (-1, -1), 0.4, _hex('#BDBDBD')),
            ('ALIGN',         (1, 0), (1, -1), 'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(sp_table)
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph(
            '5.2 Визуальное сравнение с референсными значениями', style_h2
        ))
        story.append(Paragraph(
            'Синий = в пределах нормы (отклонение <30%), '
            'красный = значимое отклонение от нормы (>30%).',
            style_note
        ))
        sp_bytes = _make_speech_bars(speech_data)
        story.append(Image(io.BytesIO(sp_bytes), width=15*cm, height=6.5*cm))
    else:
        story.append(Paragraph(
            'Речевые данные для данной записи не загружены.',
            style_note
        ))

    story.append(Spacer(1, 0.4*cm))

    if block_data:
        story.append(Paragraph('6. Анализ по блокам (Eye Tracking задания)', style_h1))
        story.append(Paragraph(
            'Пациент выполнял задания нескольких типов: PREDICTION, GAP, OVERLAP, '
            'DECISION, ANTISACCADE. Время реакции (RT) и ошибки направления '
            'анализируются раздельно по каждому блоку.',
            style_body
        ))

        blk_bytes = _make_block_rt_chart(block_data)
        story.append(Image(io.BytesIO(blk_bytes), width=15*cm, height=7*cm))
        story.append(Spacer(1, 0.3*cm))

        blk_table_data = [
            ['Блок', 'N проб', 'RT mean (мс)', 'RT median', 'Error Rate', 'Target %']
        ]
        for bd in block_data:
            blk_table_data.append([
                bd.get('block', ''),
                str(bd.get('n_trials', '')),
                f"{bd.get('rt_gaze_ms_mean') or 0:.1f}",
                f"{bd.get('rt_gaze_ms_median') or 0:.1f}",
                f"{bd.get('direction_error_rate') or 0:.3f}",
                f"{bd.get('target_reached_rate') or 0:.3f}",
            ])
        blk_table = Table(blk_table_data,
                          colWidths=[3.5*cm, 2*cm, 3*cm, 3*cm, 2.5*cm, 2.5*cm])
        blk_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), _hex('#37474F')),
            ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, _hex(C_BG)]),
            ('GRID',          (0, 0), (-1, -1), 0.4, _hex('#BDBDBD')),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(blk_table)

    # ── 7. ПОЯСНЕНИЕ К ПРИЗНАКАМ ──────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('7. Справочник признаков', style_h1))

    glossary = [
        ('MDVP:Fo(Hz)',       'Средняя частота основного тона голоса. '
                              'Снижение характерно для болезни Паркинсона.'),
        ('HNR',               'Harmonics-to-Noise Ratio. Снижение (< 20 дБ) '
                              'указывает на увеличение шума в голосе.'),
        ('Jitter / Shimmer',  'Микровариации частоты и амплитуды. '
                              'Повышение свидетельствует о нестабильности голосового тракта.'),
        ('RPDE',              'Recurrence Period Density Entropy — мера нелинейной '
                              'динамики голосового сигнала.'),
        ('DFA',               'Detrended Fluctuation Analysis — характеристика '
                              'долгосрочных корреляций в сигнале.'),
        ('PPE',               'Pitch Period Entropy — энтропия периода основного тона.'),
        ('Gaze Rate (Гц)',    'Частота записи данных айтрекера. Снижение может '
                              'указывать на качественные проблемы записи.'),
        ('Blink Rate (/мин)', 'Частота морганий. Снижение (< 10/мин) характерно '
                              'для болезни Паркинсона.'),
        ('Fixation Duration', 'Длительность фиксаций взгляда. Увеличение связано '
                              'с ухудшением когнитивного контроля.'),
        ('Saccade Amplitude', 'Амплитуда быстрых движений глаз. Снижение отражает '
                              'нарушения глазодвигательного контроля.'),
        ('RT (Reaction Time)','Время реакции в заданиях айтрекинга. '
                              'Увеличение характерно для когнитивных нарушений.'),
    ]

    for term, definition in glossary:
        story.append(Paragraph(
            f'<b>{term}</b>: {definition}', style_body
        ))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(' Важное замечание', style_h2))
    story.append(Paragraph(
        'Данный отчёт формируется автоматически системой поддержки принятия решений '
        'и носит вспомогательный характер. Результаты не являются '
        'клиническим диагнозом и не могут заменить консультацию врача-специалиста. '
        'Окончательное клиническое решение принимается лечащим врачом.',
        style_note
    ))

    doc.build(story)
    logger.info('PDF-отчёт создан: %s', output_path)
    return output_path