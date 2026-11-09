"""Motor de fusão multimodal e geração de alertas."""

from __future__ import annotations

from typing import Any
import re

from src.config import ALERT_THRESHOLDS, FUSION_WEIGHTS
from src.text.cloud_nlp import analyze_clinical_nlp, generate_clinical_report
from src.text.classifier import classify_clinical_text


def fuse_modalities(
    video_result: dict[str, Any] | None = None,
    audio_result: dict[str, Any] | None = None,
    text_input: str | None = None,
    nlp_provider: str | None = None,
    anomaly_alerts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    video_result = video_result or {}
    audio_result = audio_result or {}
    scores_v = video_result.get("scores", {})

    bleeding = float(scores_v.get("bleeding", 0.0))
    emotion = float(scores_v.get("emotion_distress", 0.0))
    pose = float(scores_v.get("pose_closed", 0.0))
    pose_result = video_result.get("pose") or {}
    pose_unavailable = pose_result.get("pose_available") is False or pose_result.get("status") == "unavailable"
    if pose_unavailable:
        pose = 0.
    audio_risk = float(audio_result.get("audio_risk_score", 0.0))

    transcript = audio_result.get("transcript") or ""
    combined_text = " ".join(t for t in [transcript, text_input or ""] if t).strip()
    text_clf = classify_clinical_text(combined_text) if combined_text else {
        "text_risk_score": 0.0,
        "top_label": "indeterminado",
        "labels": {},
    }
    cloud = analyze_clinical_nlp(combined_text, nlp_provider) if combined_text else {
        "engine": "empty", "status": "empty", "cloud_executed": False, "sentiment": None}
    text_risk = float(text_clf.get("text_risk_score", 0.0))

    final = (
        FUSION_WEIGHTS["bleeding"] * bleeding
        + FUSION_WEIGHTS["emotion_distress"] * emotion
        + FUSION_WEIGHTS["pose_closed"] * pose
        + FUSION_WEIGHTS["audio_risk"] * audio_risk
        + FUSION_WEIGHTS["text_risk"] * text_risk
    )
    final = round(min(1.0, final), 3)

    alerts: list[dict[str, Any]] = []
    if bleeding >= 0.35:
        alerts.append(
            {
                "type": "complicacao_cirurgica_visual",
                "severity": "CRITICO" if bleeding >= 0.55 else "ATENCAO",
                "message": "Possível sangramento anômalo detectado por YOLOv8/HSV.",
                "score": bleeding,
            }
        )
    if emotion >= 0.45 or pose >= 0.5:
        alerts.append(
            {
                "type": "desconforto_psicologico_visual",
                "severity": "ATENCAO",
                "message": "Indicadores visuais de desconforto (emoção/postura).",
                "score": max(emotion, pose),
            }
        )
    labels = dict(audio_result.get("labels") or {})
    for key, value in (text_clf.get("labels") or {}).items():
        labels[key] = max(labels.get(key, 0), value)
    if labels.get("violencia", 0) >= 0.35:
        alerts.append(
            {
                "type": "possivel_violencia_domestica",
                "severity": "CRITICO",
                "message": "Padrões multimodais compatíveis com risco de violência/abuso.",
                "score": max(labels.get("violencia", 0), audio_risk),
            }
        )
    if labels.get("depressao_pos_parto", 0) >= 0.35:
        alerts.append(
            {
                "type": "risco_depressao_pos_parto" if re.search(r"p[oó]s.?parto|beb[eê]|nascimento|pu[eé]rper|m[aã]e", combined_text, re.I) else "sofrimento_emocional",
                "severity": "ATENCAO",
                "message": "Sinais linguísticos/vocais em contexto de pós-parto; requer avaliação." if re.search(r"p[oó]s.?parto|beb[eê]|nascimento|pu[eé]rper|m[aã]e", combined_text, re.I) else "Sinais linguísticos de sofrimento emocional; requer avaliação contextual.",
                "score": labels.get("depressao_pos_parto", 0),
            }
        )
    if labels.get("ansiedade", 0) >= 0.35:
        alerts.append(
            {
                "type": "sinais_ansiedade",
                "severity": "ATENCAO",
                "message": "Termos de ansiedade na transcrição/texto; requer avaliação contextual.",
                "score": labels.get("ansiedade", 0),
            }
        )

    for label, alert_type in [("respiratorio", "relato_dificuldade_respiratoria"),
                              ("dificuldade_fala", "relato_dificuldade_fala"),
                              ("fadiga", "sinais_fadiga")]:
        if labels.get(label, 0) >= .35:
            alerts.append({"type": alert_type, "severity": "ATENCAO", "score": labels[label],
                           "message": "Indício lexical/vocal para revisão da equipe; não é diagnóstico."})
    alerts.extend(anomaly_alerts or [])
    # O LLM solicita revisão contextual; scores permanecem determinísticos.
    if cloud.get("engine") == "qwen-local" and cloud.get("status") == "ok":
        present = [f for f in cloud.get("findings", []) if f["assertion"] == "present"]
        if present:
            alerts.append({"type": "revisao_textual_llm", "severity": "ATENCAO",
                           "message": "Qwen identificou relatos para revisão contextual da equipe.",
                           "source": "qwen-local", "evidence": present})
    quality = []
    if pose_unavailable:
        quality.append("Postura indisponível: ausência de modelo ou de articulações visíveis; movimento global não identifica postura.")
    if audio_result and audio_result.get("status") not in {"ok", "manual", None}:
        quality.append("Áudio sem transcrição disponível; ausência de risco não foi comprovada.")
    if audio_result.get("status") == "manual":
        quality.append("Transcrição manual de demonstração; STT automático não executado.")
    if combined_text and cloud.get("status") not in {"ok", "local"}:
        quality.append(f"NLP selecionado indisponível: {cloud.get('status')}; regras locais continuam explícitas.")

    # Nível agregado: maior entre score final e severidade dos alertas tipados
    if final >= ALERT_THRESHOLDS["CRITICO"]:
        level = "CRITICO"
    elif final >= ALERT_THRESHOLDS["ATENCAO"]:
        level = "ATENCAO"
    else:
        level = "INFO"

    if any(a.get("severity") == "CRITICO" for a in alerts):
        level = "CRITICO"
    elif any(a.get("severity") == "ATENCAO" for a in alerts) and level == "INFO":
        level = "ATENCAO"

    if not alerts and level != "INFO":
        alerts.append(
            {
                "type": "risco_agregado",
                "severity": level,
                "message": "Score multimodal elevado sem alerta tipado dominante.",
                "score": final,
            }
        )

    scores = {
        "bleeding": round(bleeding, 3),
        "emotion_distress": round(emotion, 3),
        "pose_closed": round(pose, 3),
        "audio_risk": round(audio_risk, 3),
        "text_risk": round(text_risk, 3),
        "final": final,
    }

    report_ctx = {
        "alerts": alerts,
        "alert_level": level,
        "scores": scores,
        "transcript": combined_text,
        "nlp": cloud,
        "emotions": (video_result.get("emotion") or {}).get("mean_emotions"),
        "yolo_summary": {
            "frames_with_bleeding": (video_result.get("yolo") or {}).get("frames_with_bleeding"),
            "max_confidence": (video_result.get("yolo") or {}).get("max_confidence"),
            "model": (video_result.get("yolo") or {}).get("model"),
        },
    }
    report = generate_clinical_report(report_ctx)

    return {
        "assessment_status": "partial" if quality else "processed",
        "data_quality": quality,
        "alert_level": level,
        "scores": scores,
        "alerts": alerts,
        "text_classification": text_clf,
        "cloud_nlp": cloud,
        "report": report,
        "fusion_weights": FUSION_WEIGHTS,
    }
