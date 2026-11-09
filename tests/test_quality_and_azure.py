import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.cloud.azure_services import analyze_azure_text, transcribe_azure
from src.text.classifier import classify_clinical_text
from src.audio.voice_risk import analyze_audio_risk
from src.audio.transcriber import extract_audio_wav, AudioExtractionError
from src.main import main


@pytest.mark.parametrize("text", ["Paciente nega violência e ameaça.",
    "Paciente não está triste e não está chorando.", "Sem falta de ar ou dificuldade para falar."])
def test_negations_do_not_trigger_risk(text):
    result = classify_clinical_text(text)
    assert result["text_risk_score"] == 0
    assert result["negated_hits"]


def test_contrast_and_intrinsic_negation():
    assert classify_clinical_text("Nega violência, mas relata ameaça.")["labels"]["violencia"] >= .35
    assert classify_clinical_text("Não consigo dormir, sem energia.")["text_risk_score"] >= .35


def test_routine_control_is_not_violence_and_postpartum_requires_context():
    from src.fusion.multimodal_engine import fuse_modalities
    assert classify_clinical_text("Controle da pressão em consulta de rotina")["text_risk_score"] == 0
    out = fuse_modalities(text_input="Paciente triste e chorando", nlp_provider="local")
    assert "sofrimento_emocional" in {a["type"] for a in out["alerts"]}
    assert "risco_depressao_pos_parto" not in {a["type"] for a in out["alerts"]}
    assert all("pós-parto" not in a["message"] for a in out["alerts"])


def test_tone_without_transcript_is_not_anxiety(tmp_path):
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "data/samples/consulta_hesitante.wav"
    result = analyze_audio_risk(source, tmp_path, speech_provider="offline")
    assert result["audio_risk_score"] == 0
    assert "ansiedade" not in result["labels"]
    assert not result["speech_confirmed"]


def test_missing_media_not_replaced_by_silence(tmp_path):
    with pytest.raises(AudioExtractionError):
        extract_audio_wav(tmp_path / "missing.mp4", tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists()


def test_missing_text_is_indeterminate_and_bad_wav_is_unavailable(tmp_path):
    assert classify_clinical_text(None)["top_label"] == "indeterminado"
    source = tmp_path / "broken.wav"
    source.write_bytes(b"not a WAV")
    result = analyze_audio_risk(source, tmp_path / "audio", speech_provider="offline")
    assert result["status"] == "unavailable" and result["extraction_error"] == "invalid_wav"
    assert not result["speech_confirmed"]


def test_missing_azure_configuration_and_strict_cli(tmp_path):
    assert transcribe_azure(tmp_path / "a.wav")["status"] == "not_configured"
    assert analyze_azure_text("Paciente cansada")["status"] == "not_configured"
    assert main(["--audio", "x.wav", "--require-azure", "--output-dir", str(tmp_path)]) == 3


def test_text_azure_adapter_uses_service_results(monkeypatch):
    import azure.ai.textanalytics as sdk
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "test-only-key")
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    sentiment = SimpleNamespace(is_error=False, sentiment="negative", confidence_scores=SimpleNamespace(positive=.1, neutral=.2, negative=.7))
    entities = SimpleNamespace(is_error=False, entities=[SimpleNamespace(text="consulta", category="Event", confidence_score=.8)])
    phrases = SimpleNamespace(is_error=False, key_phrases=["falta de ar"])
    client = Mock()
    client.analyze_sentiment.return_value = [sentiment]
    client.recognize_entities.return_value = [entities]
    client.extract_key_phrases.return_value = [phrases]
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=client)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sdk, "TextAnalyticsClient", factory)
    result = analyze_azure_text("Relata falta de ar")
    assert result["status"] == "ok" and result["sentiment"] == "NEGATIVE"
    assert result["key_phrases"] == ["falta de ar"]
    client.analyze_sentiment.assert_called_once()
    assert "test-only-key" not in json.dumps(result)


@pytest.mark.parametrize("finish", ["ok", "canceled"])
def test_speech_continuous_collects_multiple_segments(monkeypatch, tmp_path, finish):
    import azure.cognitiveservices.speech as sdk
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-only-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "brazilsouth")

    class Signal:
        def connect(self, handler):
            self.handler = handler

    class Recognizer:
        def __init__(self, **kwargs):
            self.recognized, self.session_stopped, self.canceled = Signal(), Signal(), Signal()

        def start_continuous_recognition(self):
            for text in ["Primeira fala.", "Segunda fala."]:
                self.recognized.handler(SimpleNamespace(result=SimpleNamespace(reason=sdk.ResultReason.RecognizedSpeech,
                    text=text, offset=0, duration=10_000_000)))
            if finish == "ok":
                self.session_stopped.handler(None)
            else:
                self.canceled.handler(SimpleNamespace(reason=sdk.CancellationReason.Error))

        def stop_continuous_recognition(self):
            pass

    monkeypatch.setattr(sdk, "SpeechConfig", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(sdk.audio, "AudioConfig", lambda **kwargs: None)
    monkeypatch.setattr(sdk, "SpeechRecognizer", Recognizer)
    result = transcribe_azure(tmp_path / "fake.wav")
    assert result["status"] == ("ok" if finish == "ok" else "error")
    assert result["cloud_executed"] == (finish == "ok")
    assert len(result["segments"]) == 2
    assert result["transcript"] == "Primeira fala. Segunda fala."


def test_azure_service_failure_is_explicit_and_does_not_echo_secrets(monkeypatch):
    import azure.ai.textanalytics as sdk
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "test-only-key")
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    monkeypatch.setattr(sdk, "TextAnalyticsClient", Mock(side_effect=RuntimeError("test-only-key")))
    result = analyze_azure_text("Relata falta de ar")
    assert result["status"] == "error" and not result["cloud_executed"]
    assert "test-only-key" not in json.dumps(result)
