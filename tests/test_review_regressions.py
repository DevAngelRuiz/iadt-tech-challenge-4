"""Regressões dos achados do review: sem serviços ou downloads reais."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.fusion.multimodal_engine import fuse_modalities
from src.text.classifier import classify_clinical_text


@pytest.mark.parametrize("text,expected", [
    ("Não estou triste e ele me bate.", "violencia"),
    ("Não estou triste, ele me bate.", "violencia"),
    ("Nega violência e relata ameaça.", "violencia"),
    ("Não tenho falta de ar e estou cansada.", "fadiga"),
    ("Não estou triste e tenho medo dele.", "violencia"),
    ("Nega ansiedade, está cansada.", "fadiga"),
    ("Não estou triste e não consigo dormir.", "depressao_pos_parto"),
])
def test_negation_does_not_cross_a_new_predicate(text, expected):
    assert classify_clinical_text(text)["labels"].get(expected, 0) > 0


@pytest.mark.parametrize("text", [
    "Paciente nega violência e ameaça.",
    "Sem falta de ar e dificuldade para falar.",
    "Não estou triste e não tenho falta de ar.",
    "Não tenho medo dele e não estou cansada.",
])
def test_negated_lists_and_repeated_negation_remain_negated(text):
    assert classify_clinical_text(text)["text_risk_score"] == 0


def test_violence_alert_survives_unrelated_symptom_denial():
    result = fuse_modalities(text_input="Não estou triste e ele me bate.", nlp_provider="local")
    assert result["alert_level"] == "CRITICO"
    assert "possivel_violencia_domestica" in {a["type"] for a in result["alerts"]}


def make_video(path, asymmetric=False):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    if asymmetric:
        frame[:, 32:] = 255
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(frame)
    writer.release()
    return path


@pytest.mark.parametrize("asymmetric", [False, True])
def test_motion_fallback_does_not_infer_posture(tmp_path, asymmetric):
    from src.video.pose_analyzer import _analyze_motion_fallback
    result = _analyze_motion_fallback(make_video(tmp_path / "empty.mp4", asymmetric), 5, None)
    assert result["status"] == "unavailable" and not result["pose_available"]
    assert result["pose_closed_score"] == 0 and result["arm_movements"] is None
    # Mesmo um score antigo/inconsistente deve ser ignorado quando a postura
    # está explicitamente indisponível.
    fusion = fuse_modalities(video_result={"pose": result, "scores": {"pose_closed": .9}}, nlp_provider="local")
    assert not fusion["alerts"] and fusion["scores"]["pose_closed"] == 0
    assert fusion["assessment_status"] == "partial"


@pytest.mark.parametrize("visibility,present,available", [(1., True, True), (.1, True, False), (1., False, False)])
def test_pose_requires_detected_visible_joints(monkeypatch, tmp_path, visibility, present, available):
    import sys
    from src.video.pose_analyzer import _analyze_mediapipe_solutions
    landmarks = [SimpleNamespace(x=.5, y=.6, visibility=visibility) for _ in range(33)]
    landmarks[11].x, landmarks[12].x = .3, .7
    landmarks[11].y = landmarks[12].y = .4
    landmarks[15].x, landmarks[16].x = .45, .55
    pose = Mock()
    pose.process.return_value = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks) if present else None)
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=Mock(return_value=pose)))))
    result = _analyze_mediapipe_solutions(make_video(tmp_path / "pose.mp4"), 5, None)
    assert result["pose_available"] is available
    assert result["pose_closed_score"] == (1. if available else 0.)
    assert result["frames_sampled"] == 4
    assert result["frames_with_visible_pose"] == (4 if available else 0)
    pose.close.assert_called_once()


def azure_client(monkeypatch, labels):
    import azure.ai.textanalytics as sdk
    client = Mock()
    client.analyze_sentiment.side_effect = [[SimpleNamespace(is_error=False, sentiment=label,
        confidence_scores=SimpleNamespace(positive=.5, neutral=0., negative=.5))] for label in labels]
    client.recognize_entities.return_value = [SimpleNamespace(is_error=False, entities=[])]
    client.extract_key_phrases.return_value = [SimpleNamespace(is_error=False, key_phrases=[])]
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=client)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sdk, "TextAnalyticsClient", factory)
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "mock-only")
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://example.invalid/")
    return client


@pytest.mark.parametrize("label", ["mixed", "positive", "neutral", "negative"])
def test_azure_preserves_service_label_even_if_scores_tie(monkeypatch, label):
    from src.cloud.azure_services import analyze_azure_text
    azure_client(monkeypatch, [label])
    result = analyze_azure_text("Relato fictício.")
    assert result["status"] == "ok" and result["sentiment"] == label.upper()
    assert result["sentiment_documents"][0]["sentiment"] == label.upper()
    assert result["sentiment_aggregation"] == "service_document"


@pytest.mark.parametrize("labels,expected", [(["positive", "negative"], "MIXED"),
    (["mixed", "neutral"], "MIXED"), (["neutral", "positive"], "POSITIVE"),
    (["negative", "neutral"], "NEGATIVE"), (["neutral", "neutral"], "NEUTRAL")])
def test_azure_aggregation_keeps_each_document(monkeypatch, labels, expected):
    from src.cloud.azure_services import analyze_azure_text
    azure_client(monkeypatch, labels)
    result = analyze_azure_text("a" * 4500 + "b" * 300)
    assert result["sentiment"] == expected and result["chunks_processed"] == 2
    assert [d["offset"] for d in result["sentiment_documents"]] == [0, 4500]
    assert [d["length"] for d in result["sentiment_documents"]] == [4500, 300]
    assert [d["sentiment"] for d in result["sentiment_documents"]] == [s.upper() for s in labels]


def test_concurrent_reports_preserve_all_runs_in_same_instant(tmp_path):
    from src.reporting.report_builder import save_run_outputs
    with patch("src.reporting.report_builder.datetime") as clock:
        clock.now.return_value = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: save_run_outputs(tmp_path, {"run_id": str(i)}, "same"), range(12)))
    assert len({r["json"] for r in results}) == 12
    assert {json.loads(Path(r["json"]).read_text(encoding="utf-8"))["run_id"] for r in results} == {str(i) for i in range(12)}
    assert all(Path(r["json"]).name in Path(r["markdown"]).read_text(encoding="utf-8") for r in results)


def test_normalized_audio_files_are_not_overwritten(tmp_path):
    from src.audio.transcriber import _write_silent_wav
    from src.audio.voice_risk import analyze_audio_risk
    source = _write_silent_wav(tmp_path / "source.wav")
    work = tmp_path / "work"
    analyze_audio_risk(source, work, speech_provider="offline")
    analyze_audio_risk(source, work, speech_provider="offline")
    assert len(list(work.glob("*_risk.wav"))) == 2


def test_non_wav_audio_uses_moviepy_v2(tmp_path):
    from moviepy import AudioFileClip
    from src.audio.transcriber import extract_audio_wav
    import wave
    source = Path(__file__).resolve().parents[1] / "data/samples/consulta_hesitante.wav"
    mp3 = tmp_path / "tone.mp3"
    with AudioFileClip(str(source)) as clip:
        clip.write_audiofile(str(mp3), logger=None)
    output = extract_audio_wav(mp3, tmp_path / "normalized.wav")
    with wave.open(str(output), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() > 0


def test_qwen_verifier_loads_custom_dotenv(monkeypatch, tmp_path):
    from scripts import verify_qwen
    from src.config import load_environment
    environment = tmp_path / "test.env"
    environment.write_text("QWEN_BASE_URL=http://127.0.0.1:9999/v1\nQWEN_MODEL=test-qwen\n", encoding="utf-8")
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.setattr(verify_qwen, "ROOT", tmp_path)
    monkeypatch.setattr(verify_qwen, "load_environment", lambda: load_environment(environment))
    observed = []
    expected_by_text = {text: expected for _, text, expected in verify_qwen.CASES}
    def fake_analysis(text):
        observed.append((os.environ["QWEN_BASE_URL"], os.environ["QWEN_MODEL"]))
        return {"status": "ok", "findings": [{"category": cat, "assertion": assertion, "evidence": text}
                                                for cat, assertion in expected_by_text[text]]}
    monkeypatch.setattr(verify_qwen, "analyze_qwen_text", fake_analysis)
    assert verify_qwen.main() == 0
    assert observed == [("http://127.0.0.1:9999/v1", "test-qwen")] * 5


def test_dotenv_preserves_system_override(monkeypatch, tmp_path):
    from src.config import load_environment
    env = tmp_path / "test.env"
    env.write_text("QWEN_MODEL=file-value\n", encoding="utf-8")
    monkeypatch.setenv("QWEN_MODEL", "system-value")
    load_environment(env)
    assert os.environ["QWEN_MODEL"] == "system-value"


def test_export_uses_provider_recorded_in_evidence():
    from scripts.export_whisper_evidence import text_analysis_metadata
    payload = {"fusion": {"cloud_nlp": {"engine": "qwen-local", "status": "ok", "model": "example",
                                       "model_inference_executed": True, "cloud_executed": False},
                          "text_classification": {"model": "lexical-rules-with-negation"}}}
    result = text_analysis_metadata(payload)
    assert result["engine"] == "qwen-local" and result["model_inference_executed"]
    assert result["risk_classifier"] == "lexical-rules-with-negation"


@pytest.mark.parametrize("speech_provider", ["azure", "whisper"])
def test_cli_selects_azure_or_hybrid_without_loading_other_provider(monkeypatch, tmp_path, speech_provider):
    from src.main import main
    from src.audio.transcriber import _write_silent_wav
    for name in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.setenv(name, "mock-only")
    audio = _write_silent_wav(tmp_path / "audio.wav")
    speech_result = {"engine": "azure-speech" if speech_provider == "azure" else "whisper-local",
                     "status": "ok", "transcript": "Estou cansada.", "segments": [],
                     "cloud_executed": speech_provider == "azure",
                     "model_inference_executed": speech_provider == "whisper"}
    language_result = {"engine": "azure-text-analytics", "status": "ok", "cloud_executed": True,
                       "sentiment": "NEUTRAL", "entities": [], "key_phrases": []}
    azure_speech = Mock(return_value=speech_result)
    whisper_speech = Mock(return_value=speech_result)
    language = Mock(return_value=language_result)
    monkeypatch.setattr("src.cloud.azure_services.transcribe_azure", azure_speech)
    monkeypatch.setattr("src.audio.whisper_transcriber.transcribe_whisper", whisper_speech)
    monkeypatch.setattr("src.cloud.azure_services.analyze_azure_text", language)
    args = ["--audio", str(audio), "--speech-provider", speech_provider, "--nlp-provider", "azure",
            "--require-transcription", "--output-dir", str(tmp_path / "outputs"), "--alert-db", str(tmp_path / "alerts.sqlite")]
    if speech_provider == "azure":
        args.append("--require-azure")
    assert main(args) == 0
    assert azure_speech.call_count == (1 if speech_provider == "azure" else 0)
    assert whisper_speech.call_count == (1 if speech_provider == "whisper" else 0)
    language.assert_called_once()
    result = json.loads(next((tmp_path / "outputs").glob("run_*.json")).read_text(encoding="utf-8"))
    assert result["audio"]["stt_engine"] == speech_result["engine"]
    assert result["fusion"]["cloud_nlp"]["engine"] == "azure-text-analytics"
