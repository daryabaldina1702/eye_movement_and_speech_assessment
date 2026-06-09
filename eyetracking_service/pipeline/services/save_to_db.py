from ..models import Recording, SubjectQC, TrialMetrics, BlockSummary

def save_to_db(folder, qc, tm, bs):

    rec_obj = Recording.objects.create(
        recording_id=folder.name,
        zip_name=folder.name,
        label=0,
        label_text="unknown"
    )

    SubjectQC.objects.create(recording=rec_obj, **qc)

    for _, r in tm.iterrows():
        TrialMetrics.objects.create(recording=rec_obj, **r.to_dict())

    for _, r in bs.iterrows():
        BlockSummary.objects.create(recording=rec_obj, **r.to_dict())

    return rec_obj