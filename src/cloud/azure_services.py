"""Azure Speech e Language, com estado explícito de execução."""
from __future__ import annotations
import os
from pathlib import Path
from threading import Event


def azure_readiness() -> dict:
    return {"speech": bool(os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_SPEECH_REGION")),
            "language": bool(os.getenv("AZURE_LANGUAGE_KEY") and os.getenv("AZURE_LANGUAGE_ENDPOINT"))}


def transcribe_azure(wav_path: str | Path, language: str = "pt-BR", timeout: float = 180) -> dict:
    result = {"engine": "azure-speech", "status": "not_configured", "transcript": "",
              "segments": [], "cloud_executed": False, "wav_path": str(wav_path)}
    if not azure_readiness()["speech"]:
        return result
    try:
        import azure.cognitiveservices.speech as sdk
    except ImportError:
        return {**result, "status": "dependency_missing"}
    done, errors = Event(), []
    recognizer, started = None, False
    try:
        config = sdk.SpeechConfig(subscription=os.environ["AZURE_SPEECH_KEY"],
                                  region=os.environ["AZURE_SPEECH_REGION"])
        config.speech_recognition_language = language
        recognizer = sdk.SpeechRecognizer(speech_config=config,
                                         audio_config=sdk.audio.AudioConfig(filename=str(wav_path)))

        def recognized(event):
            if event.result.reason == sdk.ResultReason.RecognizedSpeech:
                result["segments"].append({"text": event.result.text,
                                           "offset_seconds": event.result.offset / 10_000_000,
                                           "duration_seconds": event.result.duration / 10_000_000})

        def canceled(event):
            if event.reason == sdk.CancellationReason.Error:
                errors.append("service_canceled")
            done.set()

        recognizer.recognized.connect(recognized)
        recognizer.session_stopped.connect(lambda event: done.set())
        recognizer.canceled.connect(canceled)
        recognizer.start_continuous_recognition()
        started = True
        completed = done.wait(timeout)
        recognizer.stop_continuous_recognition()
        started = False
        result["transcript"] = " ".join(s["text"] for s in result["segments"]).strip()
        result["status"] = ("timeout" if not completed else "error" if errors else
                            "ok" if result["transcript"] else "no_speech")
        result["cloud_executed"] = completed and not errors
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__)
    finally:
        if started and recognizer is not None:
            try:
                recognizer.stop_continuous_recognition()
            except Exception:
                result["status"] = "error"
    return result


def analyze_azure_text(text: str) -> dict:
    result = {"engine": "azure-text-analytics", "status": "not_configured",
              "cloud_executed": False, "sentiment": None, "sentiment_score": {},
              "entities": [], "key_phrases": []}
    if not text.strip():
        return {**result, "status": "empty"}
    if not azure_readiness()["language"]:
        return result
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        return {**result, "status": "dependency_missing"}
    try:
        chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        totals = {"positive": 0., "neutral": 0., "negative": 0.}
        documents = []
        with TextAnalyticsClient(endpoint=os.environ["AZURE_LANGUAGE_ENDPOINT"],
                                 credential=AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
                                 connection_timeout=10, read_timeout=20, retry_total=1) as client:
            for index, chunk in enumerate(chunks):
                kwargs = {"documents": [chunk], "language": "pt", "disable_service_logs": True}
                sentiment = client.analyze_sentiment(**kwargs)[0]
                entities = client.recognize_entities(**kwargs)[0]
                phrases = client.extract_key_phrases(**kwargs)[0]
                if any(doc.is_error for doc in (sentiment, entities, phrases)):
                    return {**result, "status": "error", "error_type": "DocumentError"}
                label = sentiment.sentiment.upper()
                if label not in {"POSITIVE", "NEUTRAL", "NEGATIVE", "MIXED"}:
                    raise ValueError("Invalid service sentiment")
                document_scores = {key: getattr(sentiment.confidence_scores, key) for key in totals}
                documents.append({"offset": index * 4500, "length": len(chunk),
                                  "sentiment": label, "sentiment_score": document_scores})
                for key in totals:
                    totals[key] += getattr(sentiment.confidence_scores, key) * len(chunk)
                result["entities"].extend({"text": e.text, "category": e.category,
                                           "confidence": e.confidence_score} for e in entities.entities)
                result["key_phrases"].extend(phrases.key_phrases)
        scores = {key: round(value / len(text), 4) for key, value in totals.items()}
        labels = {document["sentiment"] for document in documents}
        # Preserva o rótulo do serviço para um documento. Em múltiplos,
        # positivo + negativo ou qualquer documento misto resulta em MIXED.
        aggregate = ("MIXED" if "MIXED" in labels or {"POSITIVE", "NEGATIVE"} <= labels else
                     "POSITIVE" if "POSITIVE" in labels else "NEGATIVE" if "NEGATIVE" in labels else "NEUTRAL")
        result.update(status="ok", cloud_executed=True, sentiment=aggregate,
                      sentiment_score=scores, key_phrases=list(dict.fromkeys(result["key_phrases"])),
                      sentiment_documents=documents,
                      sentiment_aggregation="service_document" if len(documents) == 1 else "mixed_if_conflicting_or_mixed_else_non_neutral",
                      chunks_processed=len(chunks))
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__)
    return result
