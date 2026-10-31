from __future__ import annotations

import io
from datetime import datetime, timezone
from html import escape
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app_backend.services.assessment_summary import build_assessment_summary
from app_backend.services.speech_modes import speech_result_view

MODALITY_NAMES = {"tremor": "Tremor", "spiral": "Handwriting / spiral", "speech": "Speech"}


def _summary_table(summary: dict[str, Any], body_style, bold_style, note_style, accent, border) -> Table:
    rows = [[
        Paragraph("<b>Modality</b>", bold_style),
        Paragraph("<b>Step status</b>", bold_style),
        Paragraph("<b>Verified prediction</b>", bold_style),
        Paragraph("<b>Experimental observations</b>", bold_style),
        Paragraph("<b>In fusion</b>", bold_style),
    ]]
    for item in summary["modalities"]:
        probability = f"{item['probability'] * 100:.2f}%" if item["prediction_available"] else "None"
        observations = " ".join(item["observations"]) or "None saved."
        rows.append([
            Paragraph(MODALITY_NAMES[item["modality"]], body_style),
            Paragraph(str(item["step_status"]).capitalize(), body_style),
            Paragraph(f"{probability}<br/><font size=7>{item['prediction_note']}</font>", body_style),
            Paragraph(observations, note_style),
            Paragraph("Yes" if item["contributes_to_fusion"] else "No", body_style),
        ])
    table = Table(rows, colWidths=[78, 62, 150, 200, 42])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF3EA")),
            ("BOX", (0, 0), (-1, -1), 0.8, accent),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def _model_information_table(insights: dict[str, Any] | None, body_style, bold_style, note_style, accent, border):
    if not insights:
        return Paragraph("Saved experiment outputs could not be read; model validation figures are not included.", body_style)
    rows = [[
        Paragraph("<b>Modality</b>", bold_style),
        Paragraph("<b>Deployed model</b>", bold_style),
        Paragraph("<b>Pooled tuned metrics (folds 2–5)</b>", bold_style),
        Paragraph("<b>Fold ROC-AUC (mean ± SD)</b>", bold_style),
    ]]
    for modality in insights["modalities"]:
        tuned = modality["pooled_metrics"].get("tuned_svm") or {}
        fold_auc = modality["fold_metrics"].get("tuned_svm", {}).get("roc_auc", {})
        mean, sd = fold_auc.get("mean"), fold_auc.get("sd")
        metrics_text = (
            f"Accuracy {tuned.get('accuracy', 0):.3f} · Precision {tuned.get('precision', 0):.3f} · "
            f"Recall {tuned.get('recall', 0):.3f} · F1 {tuned.get('f1', 0):.3f} · ROC-AUC {tuned.get('roc_auc', 0):.3f}"
        ) if tuned else "Not available"
        auc_text = f"{mean:.3f} ± {sd:.3f}" if mean is not None and sd is not None else "Not available"
        params = modality.get("deployed_params") or {}
        param_text = ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items())
        label = MODALITY_NAMES.get(modality["modality"], modality["modality"])
        if modality["modality"] == "speech":
            label = "Speech legacy baseline (precomputed features; not used live)"
        rows.append([
            Paragraph(f"{label}<br/><font size=7>{modality['rows']} rows · {modality['participant_groups']} participant groups</font>", body_style),
            Paragraph(f"{modality['deployed_family'] or 'n/a'} (calibrated) · {param_text}", note_style),
            Paragraph(metrics_text, note_style),
            Paragraph(auc_text, body_style),
        ])
    speech_v2 = insights["speech_audio_v2"]
    reading = speech_v2["modes"].get("reading", {})
    if reading.get("model_trained"):
        speech_auc = reading.get("outer_fold_roc_auc_mean")
        speech_sd = reading.get("outer_fold_roc_auc_sd")
        rows.append([
            Paragraph("Speech reading passage (speech_audio_v2)<br/><font size=7>EXPERIMENTAL</font>", body_style),
            Paragraph(f"{reading.get('deployed_family') or 'n/a'} (calibrated) · 44 features", note_style),
            Paragraph(
                f"{reading.get('subjects')} participants ({reading.get('control_subjects')} control, "
                f"{reading.get('parkinson_subjects')} PD); majority-class accuracy {reading.get('majority_class_accuracy', 0):.3f}. "
                "Subject-disjoint validation; not clinically validated.", note_style),
            Paragraph(f"{speech_auc:.3f} ± {speech_sd:.3f}" if speech_auc is not None and speech_sd is not None else "Not available", body_style),
        ])
    else:
        rows.append([
            Paragraph("Speech (speech_audio_v2)", body_style),
            Paragraph("No trained model", body_style),
            Paragraph(speech_v2.get("blocker") or "No trained speech_audio_v2 model.", note_style),
            Paragraph("Not available", body_style),
        ])
    table = Table(rows, colWidths=[118, 150, 170, 94])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF3EA")),
            ("BOX", (0, 0), (-1, -1), 0.8, accent),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def generate_assessment_pdf(
    assessment: dict[str, Any],
    user_info: dict[str, Any] | None = None,
    insights: dict[str, Any] | None = None,
) -> bytes:
    """Generate a readable, professional PDF research report for an assessment.

    ``insights`` is the saved research output from ``build_research_insights``.
    It is optional so the report still renders when the experiment files are absent.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom color palette aligned with the sage theme
    primary_color = colors.HexColor("#3E5438")  # Dark sage green
    accent_color = colors.HexColor("#526B4A")   # Deep forest sage
    muted_sage = colors.HexColor("#788B6A")     # Muted sage
    text_dark = colors.HexColor("#29352C")      # Dark readable text
    bg_warm = colors.HexColor("#F6F5F0")        # Warm off-white
    border_color = colors.HexColor("#D0D7C9")   # Subtle border
    warning_bg = colors.HexColor("#FFF8E7")     # Light warning box
    warning_border = colors.HexColor("#D4A574")

    # Typography styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=muted_sage,
    )

    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=text_dark,
    )

    bold_body_style = ParagraphStyle(
        "BodyDarkBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=text_dark,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#7C2D12"),
    )

    small_note_style = ParagraphStyle(
        "SmallNote",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4A5A4A"),
    )

    story: list[Any] = []

    # Title & Header
    story.append(Paragraph("Parkinson's Disease Research Assessment Report", title_style))
    story.append(Paragraph("Multimodal Research Prototype — Final Evaluation Record", subtitle_style))
    story.append(Spacer(1, 10))

    # Prominent Disclaimer Box
    disclaimer_text = (
        "<b>RESEARCH PROTOTYPE NOTICE:</b> This report was generated by an academic research "
        "prototype for algorithmic study. It is <b>NOT</b> a medical diagnostic tool or clinical assessment. "
        "Model outputs represent estimated class probabilities from machine learning models trained on "
        "benchmark datasets and must never be used as a diagnosis or substitute for assessment by a qualified clinician."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[532],
    )
    disclaimer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), warning_bg),
            ("BOX", (0, 0), (-1, -1), 1, warning_border),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(disclaimer_table)
    story.append(Spacer(1, 12))

    # Metadata Summary Table
    assessment_id = assessment.get("id") or assessment.get("assessment_id", "N/A")
    completed_at = assessment.get("completed_at") or assessment.get("updated_at") or "In Progress"
    user_ref = f"Participant #{user_info.get('id', assessment.get('user_id', 'ANON'))}" if user_info or "user_id" in assessment else "Anonymous Subject"
    status_str = str(assessment.get("status", "unknown")).upper()

    meta_data = [
        [
            Paragraph("<b>Assessment ID:</b>", body_style),
            Paragraph(f"PRJ-PD-{assessment_id}", body_style),
            Paragraph("<b>Participant Ref:</b>", body_style),
            Paragraph(user_ref, body_style),
        ],
        [
            Paragraph("<b>Assessment Date:</b>", body_style),
            Paragraph(str(completed_at), body_style),
            Paragraph("<b>Lifecycle Status:</b>", body_style),
            Paragraph(status_str, bold_body_style),
        ],
        [
            Paragraph("<b>Report Generated:</b>", body_style),
            Paragraph(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), body_style),
            Paragraph("<b>Report Version:</b>", body_style),
            Paragraph("1.0.0 (Calibrated SVM / RF)", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 156, 110, 156])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg_warm),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Assessment summary: modality status, verified predictions, observations and fusion scope
    steps = assessment.get("steps", {})
    summary = build_assessment_summary(assessment)
    story.append(Paragraph("Assessment Summary, Modality Status & Fusion Scope", h2_style))
    story.append(_summary_table(summary, body_style, bold_body_style, small_note_style, accent_color, border_color))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Fusion status:</b> {summary['fusion']['note']}", body_style))
    fusion = (assessment.get("results") or {}).get("fusion")
    if summary["fusion"]["fused_probability_displayed"] and fusion:
        prob_val = fusion.get("probability")
        threshold = fusion.get("threshold", 0.5)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Single-model probability:</b> {prob_val * 100:.2f}% (threshold {threshold:.2f}). "
            "This is one modality's classifier output, not a combined multimodal result.",
            body_style,
        ))
    elif fusion and fusion.get("effective_weights"):
        weights = ", ".join(f"{name}: {value * 100:.0f}%" for name, value in fusion["effective_weights"].items())
        story.append(Paragraph(
            f"<b>Contribution weights (AUC-normalized, not probabilities):</b> {weights}. "
            "The combined probability is withheld because the modality cohorts are unmatched.",
            small_note_style,
        ))

    story.append(Spacer(1, 14))

    # Modality 1: Tremor Assessment
    story.append(Paragraph("1. Tremor Assessment (UPDRS III Motor Ratings)", h2_style))
    tremor_step = steps.get("tremor", {})
    t_status = tremor_step.get("status", "pending")
    t_outcome = tremor_step.get("outcome", {})

    tremor_items_map = [
        ("updrs_iii_motor_scale_specific_items_20_tremor_at_rest_head", "Item 20: Rest Tremor — Head (face, lips, chin)"),
        ("updrs_iii_motor_scale_specific_items_20_tremor_at_rest_rue", "Item 20: Rest Tremor — Right Upper Extremity (RUE)"),
        ("updrs_iii_motor_scale_specific_items_20_tremor_at_rest_lue", "Item 20: Rest Tremor — Left Upper Extremity (LUE)"),
        ("updrs_iii_motor_scale_specific_items_20_tremor_at_rest_rle", "Item 20: Rest Tremor — Right Lower Extremity (RLE)"),
        ("updrs_iii_motor_scale_specific_items_20_tremor_at_rest_lle", "Item 20: Rest Tremor — Left Lower Extremity (LLE)"),
        ("updrs_iii_motor_scale_specific_items_21_action_or_postural_tremor_rue", "Item 21: Action/Postural Tremor — Right Upper Extremity"),
        ("updrs_iii_motor_scale_specific_items_21_action_or_postural_tremor_lue", "Item 21: Action/Postural Tremor — Left Upper Extremity"),
    ]

    t_data_rows = [
        [
            Paragraph("<b>UPDRS III Item / Clinical Location</b>", bold_body_style),
            Paragraph("<b>Entered Rating (0–4)</b>", bold_body_style),
            Paragraph("<b>Clinical Severity Meaning</b>", bold_body_style),
        ]
    ]

    scale_meanings = {
        0: "0 — None (Normal)",
        1: "1 — Slight (infrequently present)",
        2: "2 — Mild (persistent) / Moderate (intermittent)",
        3: "3 — Moderate (present most of time)",
        4: "4 — Extreme (present most of time)",
    }

    t_inputs = t_outcome.get("input_features", {}) if isinstance(t_outcome, dict) else {}
    for code, label in tremor_items_map:
        val = t_inputs.get(code)
        val_str = str(int(val)) if val is not None else "—"
        meaning = scale_meanings.get(int(val), "Not provided") if val is not None else "—"
        t_data_rows.append([
            Paragraph(label, body_style),
            Paragraph(val_str, body_style),
            Paragraph(meaning, body_style),
        ])

    t_table = Table(t_data_rows, colWidths=[240, 110, 182])
    t_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), bg_warm),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(t_table)

    if t_outcome and "probability" in t_outcome:
        t_prob = t_outcome["probability"]
        t_pred = "Parkinson's Class" if t_outcome.get("prediction") == 1 else "Control Class"
        t_meta_text = (
            f"<b>Model Result:</b> {t_pred} (Probability: <b>{t_prob * 100:.2f}%</b>) | "
            f"<b>Model Artifact:</b> <code>tremor_svm_tuned.joblib</code> (Calibrated RBF SVM, C=1.0, gamma=0.1) | "
            f"<b>Test ROC-AUC:</b> 0.8997"
        )
        story.append(Spacer(1, 4))
        story.append(Paragraph(t_meta_text, body_style))
    else:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Tremor Status: {t_status.capitalize()}", small_note_style))

    story.append(Spacer(1, 14))

    # Modality 2: Handwriting / Spiral Assessment
    story.append(Paragraph("2. Handwriting & Spiral Test (Trajectory & Kinematics)", h2_style))
    spiral_step = steps.get("spiral", {})
    sp_status = spiral_step.get("status", "pending")
    sp_outcome = spiral_step.get("outcome", {})
    sp_error = spiral_step.get("error")

    canvas_observation = sp_outcome.get("canvas_telemetry", {}) if isinstance(sp_outcome, dict) else {}
    if canvas_observation:
        pressure_coverage = canvas_observation.get("pressure_coverage")
        pressure_text = "unavailable" if pressure_coverage is None else f"{pressure_coverage * 100:.1f}% of samples"
        canvas_lines = [
            "<b>Saved 2D canvas telemetry (not used by the digitizer-trained classifier):</b>",
            f"Samples: {canvas_observation.get('sample_count', 'N/A')} | Strokes: {canvas_observation.get('stroke_count', 'N/A')} | "
            f"Path length: {canvas_observation.get('path_length_px', 'N/A')} px | Mean speed: {canvas_observation.get('mean_speed_px_per_second', 'N/A')} px/s",
            f"Mean acceleration: {canvas_observation.get('mean_acceleration_px_per_second2', 'N/A')} px/s^2 | "
            f"Mean jerk: {canvas_observation.get('mean_jerk_px_per_second3', 'N/A')} px/s^3 | Pressure reported by device: {pressure_text}",
            "<i>These canvas observations are screen-coordinate measurements. They do not provide calibrated pressure, grip angle, Z, or digitizer-equivalent measurements.</i>",
        ]
        story.append(Paragraph("<br/>".join(canvas_lines), small_note_style))

    if sp_outcome and "probability" in sp_outcome:
        sp_prob = sp_outcome["probability"]
        sp_pred = "Parkinson's Class" if sp_outcome.get("prediction") == 1 else "Control Class"
        sp_text = (
            f"<b>Model Result:</b> {sp_pred} (Probability: <b>{sp_prob * 100:.2f}%</b>)<br/>"
            f"<b>Input Type:</b> Digitizer / Pointer Telemetry Trajectory (PaHaW Schema)<br/>"
            f"<b>Model Artifact:</b> <code>handwriting_svm_tuned.joblib</code> (Calibrated RBF SVM, C=0.1, gamma=0.01) | "
            f"<b>Test ROC-AUC:</b> 0.9322"
        )
        story.append(Paragraph(sp_text, body_style))
    elif sp_outcome and sp_outcome.get("visual_analysis"):
        va = sp_outcome["visual_analysis"]
        va_text = (
            f"<b>Visual Inspection Analysis (OpenCV Video Proxy):</b><br/>"
            f"• Detected Spiral Center: ({va.get('center_x', 'N/A')}, {va.get('center_y', 'N/A')})<br/>"
            f"• Trace Contour Perimeter: {va.get('perimeter', 'N/A')} px | Irregularity Metric: {va.get('irregularity', 'N/A')}<br/>"
            f"• Optical Flow Oscillation Index: {va.get('oscillation_index', 'N/A')}<br/>"
            f"<i>Notice: Visual inspection is an image-based proxy and cannot measure tablet pen pressure or grip angle.</i>"
        )
        story.append(Paragraph(va_text, body_style))
    else:
        status_desc = (
            f"<b>Status:</b> {sp_status.capitalize()}. "
            f"{sp_error or 'Camera video cannot directly measure tablet pressure or grip angle. Trajectory telemetry required for classifier inference.'}"
        )
        story.append(Paragraph(status_desc, body_style))

    video_analysis = sp_outcome.get("video_analysis", {}) if isinstance(sp_outcome, dict) else {}
    if video_analysis:
        video_lines = [
            "<b>Separate spiral video analysis (image measurements from OpenCV and MediaPipe; no classifier used):</b>",
            f"Frames: {video_analysis.get('video', {}).get('frame_count', 'N/A')} | Duration: {video_analysis.get('video', {}).get('duration_seconds', 'N/A')} s",
        ]
        for key, fallback in (("ink_and_strokes", "Ink and strokes"), ("hand_and_grip", "Hand and grip"),
                              ("spiral_geometry", "Spiral geometry")):
            section = video_analysis.get("sections", {}).get(key, {})
            items = section.get("items", [])
            values = "; ".join(
                f"{escape(str(item.get('label', '')))}: {escape(str(item.get('value', '')))}"
                + (f" {escape(str(item.get('unit')))}" if item.get("unit") else "")
                for item in items
            )
            video_lines.append(f"<b>{escape(section.get('title', fallback))}:</b> {values or 'no values measured'}")
        video_lines.append("Annotated video is available through authenticated assessment history.")
        video_lines.extend(f"<i>Limitation: {escape(str(item))}</i>" for item in video_analysis.get("limitations", []))
        story.append(Paragraph("<br/>".join(video_lines), small_note_style))

    story.append(Spacer(1, 14))

    # Modality 3: Speech Assessment
    story.append(Paragraph("3. Speech Assessment (Acoustic Dysphonia Features)", h2_style))
    speech_step = steps.get("speech", {})
    spch_status = speech_step.get("status", "pending")
    spch_outcome = speech_step.get("outcome", {})
    spch_error = speech_step.get("error")

    # The three states are reported separately: extraction, prediction and the reason a prediction is missing.
    view = speech_result_view(speech_step)
    if view["recording_mode"] == "reading":
        passage_text = (
            "<b>Standardized Reading Passage:</b><br/>"
            "<i>\"Today is a pleasant day. I am speaking at my normal pace and volume. "
            "I will read each sentence clearly and naturally, without rushing. "
            "After finishing the passage, I will pause briefly before ending the recording.\"</i>"
        )
        story.append(Paragraph(passage_text, body_style))
    story.append(Paragraph(f"<b>Recording task:</b> {escape(view['recording_mode_label'] or 'not recorded')}", body_style))
    story.append(Spacer(1, 4))

    if view["feature_extraction"] == "completed":
        extraction_text = (
            f"<b>Feature extraction:</b> completed, {view['feature_count']} {escape(str(view['feature_schema']))} "
            "features (no audio file was retained)."
        )
    elif view["feature_extraction"] == "failed":
        extraction_text = f"<b>Feature extraction:</b> failed. {escape(str(spch_error or ''))}"
    else:
        extraction_text = "<b>Feature extraction:</b> not run."
    story.append(Paragraph(extraction_text, body_style))

    if view["prediction"] == "completed":
        predicted_class = "Parkinson's class" if view["prediction_value"] == 1 else "Control class"
        prediction_text = (
            f"<b>Prediction:</b> {predicted_class} "
            f"(probability <b>{view['probability'] * 100:.2f}%</b>) from the {escape(view['model'] or '')}. "
            "Research output only; validation is reported in Section 6."
        )
    elif view["prediction"] == "unavailable":
        prediction_text = f"<b>Prediction:</b> unavailable. {escape(view['prediction_reason'] or '')}"
    else:
        prediction_text = f"<b>Prediction:</b> not run. {escape(view['prediction_reason'] or '')}"
    story.append(Paragraph(prediction_text, body_style))
    story.append(Spacer(1, 14))

    # Section 4: Telemonitoring Severity Regression Limitation Note
    story.append(Paragraph("4. Severity Estimation (Motor UPDRS Telemonitoring)", h2_style))
    severity_note = (
        "<b>Motor UPDRS Severity Regression Status: DISABLED / EXPERIMENTAL</b><br/>"
        "In offline grouped 5-fold cross-validation on the 42-subject Oxford Telemonitoring Dataset, "
        "the trained regressors achieved negative R² scores (Ridge Regression R² = -0.250, Random Forest R² = -0.466), "
        "performing worse than the participant group mean baseline (R² = -0.047). "
        "In accordance with rigorous research standards, regression severity scoring is not presented as a reliable "
        "metric in clinical assessment reports."
    )
    story.append(Paragraph(severity_note, body_style))
    story.append(Spacer(1, 14))

    # Section 5: Methodology Summary
    story.append(Paragraph("5. Technical Methodology Summary", h2_style))
    method_text = (
        "<b>Cross-Validation:</b> StratifiedGroupKFold (seed 42) ensuring all recordings from any participant "
        "remain strictly within the same fold to prevent data leakage.<br/>"
        "<b>Pre-processing:</b> Median imputation and standard scaling fitted strictly within training partitions.<br/>"
        "<b>Probability Calibration:</b> Sigmoid Platt scaling fitted via inner grouped cross-validation.<br/>"
        "<b>Late-Fusion Formulation:</b> Normalized validation-fold ROC-AUC weights: "
        "Speech (w = 0.350142), Handwriting (w = 0.336668), Tremor (w = 0.313190)."
    )
    story.append(Paragraph(method_text, small_note_style))
    story.append(Spacer(1, 16))

    # Section 6: Model information and saved validation results
    story.append(Paragraph("6. Model Information & Saved Validation Results", h2_style))
    story.append(_model_information_table(insights, body_style, bold_body_style, small_note_style, accent_color, border_color))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Metrics are grouped five-fold participant-level cross-validation estimates on small cohorts (folds 2–5 reported; "
        "fold 1 reserved for fusion weights). They are research estimates, not clinical validation.",
        small_note_style,
    ))
    story.append(Spacer(1, 12))

    # Section 7: Limitations
    story.append(Paragraph("7. Limitations & Scope", h2_style))
    limitation_items = summary["limitations"]
    story.append(Paragraph("<br/>".join(f"&bull; {item}" for item in limitation_items), body_style))
    story.append(Spacer(1, 14))

    # Footer line
    story.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceAfter=8))
    footer_text = (
        "Parkinson's Disease Research Project | Department of Computer Science & Engineering | "
        "Research report | Experimental result, not a diagnosis"
    )
    story.append(Paragraph(footer_text, small_note_style))

    doc.build(story)
    return buffer.getvalue()
