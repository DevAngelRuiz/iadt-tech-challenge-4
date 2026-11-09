"""NLP com provedor explícito e relatório local por padrão."""

from __future__ import annotations

import os
from typing import Any

from src.text.classifier import classify_clinical_text
from src.text.summarizer import summarize_text


def analyze_clinical_nlp(text: str, provider: str | None = None) -> dict[str, Any]:
    provider = provider or os.getenv("NLP_PROVIDER", "local")
    if provider == "azure":
        from src.cloud.azure_services import analyze_azure_text
        result = analyze_azure_text(text)
    elif provider == "aws":
        result = analyze_with_comprehend(text)
        result.update(status="ok" if result["engine"] == "aws-comprehend" else "unavailable",
                      cloud_executed=result["engine"] == "aws-comprehend")
    elif provider == "local":
        clf = classify_clinical_text(text)
        result = {"engine": "local-rules", "status": "local", "cloud_executed": False,
                  "sentiment": None, "sentiment_score": {}, "entities": [],
                  "key_phrases": sorted({h for values in clf["hits"].values() for h in values})}
    elif provider == "qwen":
        from src.text.qwen_nlp import analyze_qwen_text
        result = analyze_qwen_text(text)
    else:
        raise ValueError("Provedor de NLP inválido")
    # Termos críticos são regras locais sobre o texto. Sentimento não é diagnóstico.
    result["critical_terms"] = classify_clinical_text(text)["hits"]
    result["critical_terms_model"] = "local-lexical-rules"
    return result


def analyze_with_comprehend(text: str) -> dict[str, Any]:
    """Análise de sentimento/entidades via AWS Comprehend ou fallback local."""
    text = (text or "").strip()
    if not text:
        return {"engine": "empty", "sentiment": "NEUTRAL", "sentiment_score": {}, "entities": [], "key_phrases": []}

    access = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if access and secret:
        try:
            import boto3

            client = boto3.client("comprehend", region_name=region)
            lang = "pt"
            sentiment = client.detect_sentiment(Text=text[:4500], LanguageCode=lang)
            entities = client.detect_entities(Text=text[:4500], LanguageCode=lang)
            phrases = client.detect_key_phrases(Text=text[:4500], LanguageCode=lang)
            return {
                "engine": "aws-comprehend",
                "sentiment": sentiment.get("Sentiment"),
                "sentiment_score": sentiment.get("SentimentScore", {}),
                "entities": entities.get("Entities", [])[:20],
                "key_phrases": [p.get("Text") for p in phrases.get("KeyPhrases", [])[:20]],
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Comprehend falhou: {exc}")

    # Fallback local
    clf = classify_clinical_text(text)
    sentiment = "NEGATIVE" if clf["text_risk_score"] >= 0.4 else "NEUTRAL"
    if clf["top_label"] == "normal":
        sentiment = "NEUTRAL"
    return {
        "engine": "local-fallback",
        "sentiment": sentiment,
        "sentiment_score": {
            "Negative": clf["text_risk_score"],
            "Neutral": 1.0 - clf["text_risk_score"],
            "Positive": 0.0,
            "Mixed": 0.0,
        },
        "entities": [],
        "key_phrases": list({h for hits in clf.get("hits", {}).values() for h in hits})[:20],
        "local_classification": clf,
    }


def generate_clinical_report(context: dict[str, Any]) -> dict[str, Any]:
    """Gera relatório clínico consolidado com OpenAI ou template local."""
    blob = (
        f"Alertas: {context.get('alerts')}\n"
        f"Nível: {context.get('alert_level')}\n"
        f"Scores: {context.get('scores')}\n"
        f"Transcrição: {context.get('transcript')}\n"
        f"Emoções: {context.get('emotions')}\n"
        f"YOLO: {context.get('yolo_summary')}\n"
    )
    summary = summarize_text(blob) if os.getenv("REPORT_PROVIDER", "local") == "openai" else {}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and summary.get("engine") == "openai-gpt":
        return {"report": summary["summary"], "engine": "openai-gpt"}

    # Template local estruturado
    level = context.get("alert_level", "INFO")
    alerts = context.get("alerts", [])
    lines = [
        "# Relatório clínico multimodal — Saúde da Mulher",
        "",
        f"**Nível de alerta:** {level}",
        "",
        "## Achados",
    ]
    if alerts:
        for a in alerts:
            lines.append(f"- [{a.get('severity')}] {a.get('type')}: {a.get('message')}")
    else:
        lines.append("- Nenhum alerta crítico gerado.")
    lines.extend(
        [
            "",
            "## Scores",
            f"- Sangramento (YOLO): {context.get('scores', {}).get('bleeding', 0)}",
            f"- Distresse emocional: {context.get('scores', {}).get('emotion_distress', 0)}",
            f"- Postura fechada: {context.get('scores', {}).get('pose_closed', 0)}",
            f"- Risco áudio: {context.get('scores', {}).get('audio_risk', 0)}",
            f"- Risco texto: {context.get('scores', {}).get('text_risk', 0)}",
            f"- Score final: {context.get('scores', {}).get('final', 0)}",
            "",
            "## Transcrição (trecho)",
            (context.get("transcript") or "_sem áudio/texto_")[:500],
            "",
            "## Recomendação",
            "Encaminhar para avaliação da equipe especializada conforme protocolo institucional."
            if level in {"ATENCAO", "CRITICO"}
            else "Manter monitoramento de rotina.",
        ]
    )
    nlp = context.get("nlp") or {}
    if nlp.get("engine") == "qwen-local":
        lines.extend(["", "## Análise textual — Qwen local", "",
                      f"Estado: {nlp.get('status')}; modelo: {nlp.get('model')}",
                      "Scores de texto permanecem baseados em regras locais."])
        if nlp.get("status") == "ok":
            lines.extend(["Resumo gerado (requer revisão): " + nlp["summary"],
                          "Sentimento textual (não é gravidade): " + nlp["sentiment"]])
            lines.extend(f"- {f['category']} / {f['assertion']}: {f['evidence']}" for f in nlp["findings"])
    return {"report": "\n".join(lines), "engine": "local-template"}
