import json
import sys
from types import SimpleNamespace
import wave
import numpy as np
import pytest
from src.audio import whisper_transcriber as adapter
from src.main import main


@pytest.fixture
def spoken_input(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.full(16000, 5000, dtype="<i2").tobytes())
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace())
    return path


def test_array_transcription_in_portuguese_without_ffmpeg(spoken_input, monkeypatch):
    def decode(audio, **kwargs):
        assert audio.dtype == np.float32 and audio.shape == (16000,)
        assert kwargs["language"] == "pt" and kwargs["fp16"] is False
        return {"segments": [{"text": " Estou cansada. ", "start": 0., "end": 1.,
                              "no_speech_prob": .01, "avg_logprob": -.1, "compression_ratio": 1.}]}
    monkeypatch.setattr(adapter, "_load_model", lambda *args: SimpleNamespace(transcribe=decode))
    result = adapter.transcribe_whisper(spoken_input)
    assert result["transcript"] == "Estou cansada."
    assert result["status"] == "ok" and result["model_inference_executed"]
    assert not result["cloud_executed"]
    assert result["engine"] == "whisper-local"


@pytest.mark.parametrize("quality", [{"no_speech_prob": .9}, {"avg_logprob": -2.}, {"compression_ratio": 4.}])
def test_unreliable_segments_are_not_presented_as_speech(spoken_input, monkeypatch, quality):
    segment = {"text": "Estou cansada", "start": 0., "end": 1., **quality}
    monkeypatch.setattr(adapter, "_load_model", lambda *args: SimpleNamespace(transcribe=lambda *a, **k: {"segments": [segment]}))
    result = adapter.transcribe_whisper(spoken_input)
    assert result["status"] == "no_speech" and not result["transcript"]
    assert result["model_inference_executed"]


def test_silence_does_not_download_or_load_a_model(tmp_path, monkeypatch):
    from src.audio.transcriber import _write_silent_wav
    path = _write_silent_wav(tmp_path / "silence.wav")
    def forbidden(*args):
        raise AssertionError("Modelo não deveria ser carregado")
    monkeypatch.setattr(adapter, "_load_model", forbidden)
    result = adapter.transcribe_whisper(path)
    assert result["status"] == "no_speech" and not result["model_inference_executed"]


def test_missing_package_is_explicit(spoken_input, monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)
    assert adapter.transcribe_whisper(spoken_input)["status"] == "dependency_missing"


def test_model_loading_failure_is_explicit(spoken_input, monkeypatch):
    def broken(*args):
        raise OSError("falha de download ou espaço")
    monkeypatch.setattr(adapter, "_load_model", broken)
    result = adapter.transcribe_whisper(spoken_input)
    assert result["status"] == "error" and not result["model_inference_executed"]


def test_strict_mode_rejects_manual_and_missing_audio(tmp_path):
    assert main(["--audio", "x.wav", "--transcript-override", "fala manual", "--require-transcription",
                 "--output-dir", str(tmp_path)]) == 3
    assert main(["--audio", str(tmp_path / "missing.wav"), "--speech-provider", "whisper", "--nlp-provider", "local",
                 "--require-transcription", "--output-dir", str(tmp_path), "--alert-db", str(tmp_path / "alerts.sqlite")]) == 3
    result = json.loads(next(tmp_path.glob("run_*.json")).read_text(encoding="utf-8"))
    assert result["audio"]["status"] == "unavailable"
