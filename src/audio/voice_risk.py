"""Análise de risco vocal/clínico a partir de áudio e transcrição."""

from __future__ import annotations

import wave
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.audio.transcriber import AudioExtractionError, extract_audio_wav, transcribe_wav
from src.config import RISK_LEXICON
from src.text.classifier import classify_clinical_text


def _prosody_features(wav_path: Path) -> dict[str, float]:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        width = wf.getsampwidth()

    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0

    if samples.size == 0:
        return {"duration_sec": 0.0, "rms": 0.0, "silence_ratio": 1.0, "pause_score": 1.0, "fatigue_score": 0.0}

    samples = samples / (32768.0 if width == 2 else 128.0)
    duration = samples.size / float(rate)
    frame = max(1, int(0.02 * rate))
    energies = []
    for i in range(0, samples.size, frame):
        chunk = samples[i : i + frame]
        energies.append(float(np.sqrt(np.mean(chunk**2))))
    energies_arr = np.array(energies) if energies else np.array([0.0])
    rms = float(np.mean(energies_arr))
    silence_ratio = float(np.mean(energies_arr < 0.02))
    # Pausas longas → hesitação
    pause_score = min(1.0, silence_ratio * 1.2)
    # Baixa energia → fadiga
    fatigue_score = min(1.0, max(0.0, 0.6 - rms) / 0.6)
    return {
        "duration_sec": round(duration, 2),
        "rms": round(rms, 4),
        "silence_ratio": round(silence_ratio, 3),
        "pause_score": round(pause_score, 3),
        "fatigue_score": round(fatigue_score, 3),
    }


def analyze_audio_risk(
    media_path: str | Path,
    work_dir: str | Path,
    transcript_override: str | None = None,
    speech_provider: str | None = None,
    language: str = "pt-BR",
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{Path(media_path).stem}_{uuid.uuid4().hex}_risk.wav"
    extraction_error = None
    try:
        extract_audio_wav(media_path, wav_path)
        prosody = _prosody_features(wav_path)
    except AudioExtractionError as exc:
        extraction_error = str(exc)
        prosody = {"duration_sec": 0, "rms": 0, "pause_score": 0, "fatigue_score": 0}

    if transcript_override is not None:
        transcript = transcript_override
        engine = "override"
        tr = {"status": "manual", "cloud_executed": False}
    elif extraction_error:
        transcript, engine = "", "none"
        tr = {"status": "unavailable", "cloud_executed": False}
    else:
        tr = transcribe_wav(wav_path, provider=speech_provider, language=language)
        transcript = tr["transcript"]
        engine = tr["engine"]

    # Se STT falhou (áudio sintético sem fala), permite texto demo acoplado
    classification = classify_clinical_text(transcript) if transcript else {
        "labels": {},
        "top_label": "indeterminado",
        "text_risk_score": 0.0,
    }

    # Combina léxico + prosódia (prosódia só reforça se houver sinal acústico real)
    label_scores = dict(classification.get("labels", {}))
    acoustic_ok = (tr["status"] == "ok" and bool(transcript.strip())
                   and prosody["rms"] > 0.01 and prosody["duration_sec"] > 0.5)
    if acoustic_ok and prosody["fatigue_score"] > 0.4 and label_scores.get("fadiga", 0) > 0:
        label_scores["fadiga"] = max(label_scores.get("fadiga", 0.0), prosody["fatigue_score"])
    # Pausas sozinhas não identificam ansiedade ou contexto gestacional.

    text_component = classification.get("text_risk_score", 0.0)
    if acoustic_ok:
        audio_risk = min(
            1.0,
            0.6 * text_component + 0.25 * prosody["pause_score"] + 0.15 * prosody["fatigue_score"],
        )
    else:
        audio_risk = text_component


    return {
        "modality": "audio",
        "source": str(media_path),
        "transcript": transcript,
        "stt_engine": engine,
        "status": tr["status"],
        "cloud_executed": tr.get("cloud_executed", False),
        "model_inference_executed": tr.get("model_inference_executed", False),
        "stt_model": tr.get("model"),
        "stt_device": tr.get("device"),
        "segments": tr.get("segments", []),
        "extraction_error": extraction_error,
        "speech_confirmed": acoustic_ok,
        "vocal_findings": (["pausas_frequentes"] if acoustic_ok and prosody["pause_score"] > .5 else [])
                          + (["baixa_energia_vocal"] if acoustic_ok and prosody["fatigue_score"] > .4 else []),
        "prosody": prosody,
        "labels": {k: round(v, 3) for k, v in label_scores.items()},
        "top_label": max(label_scores, key=label_scores.get) if label_scores else "indeterminado",
        "audio_risk_score": round(audio_risk, 3),
        "lexicon_keys": list(RISK_LEXICON.keys()),
    }
