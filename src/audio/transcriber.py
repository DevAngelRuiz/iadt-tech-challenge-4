"""Extração verificável e transcrição com provedor explícito."""
from __future__ import annotations
import os
import wave
from pathlib import Path
import numpy as np


class AudioExtractionError(ValueError):
    pass


def _normalize_pcm(source: Path, target: Path) -> Path:
    with wave.open(str(source), "rb") as stream:
        channels, width, rate = stream.getnchannels(), stream.getsampwidth(), stream.getframerate()
        raw = stream.readframes(stream.getnframes())
    if width != 2:
        raise AudioExtractionError("unsupported_pcm_width")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, channels).mean(axis=1)
    if rate != 16000 and samples.size:
        samples = np.interp(np.arange(round(len(samples) * 16000 / rate)) * rate / 16000,
                            np.arange(len(samples)), samples)
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())
    return target


def extract_audio_wav(media_path: str | Path, wav_path: str | Path) -> Path:
    source, target = Path(media_path), Path(wav_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise AudioExtractionError("source_not_found")
    if source.suffix.lower() == ".wav":
        try:
            return _normalize_pcm(source, target)
        except (wave.Error, EOFError, ValueError, OSError) as exc:
            raise AudioExtractionError("invalid_wav") from exc
    clip = audio = None
    tmp = target.with_suffix(".extract.wav")
    try:
        from moviepy import AudioFileClip, VideoFileClip
        if source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            clip = VideoFileClip(str(source))
            audio = clip.audio
        else:
            audio = AudioFileClip(str(source))
        if audio is None:
            raise AudioExtractionError("no_audio_track")
        audio.write_audiofile(str(tmp), fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
        return _normalize_pcm(tmp, target)
    except AudioExtractionError:
        raise
    except Exception as exc:
        raise AudioExtractionError("extraction_failed") from exc
    finally:
        if audio is not None:
            audio.close()
        if clip is not None:
            clip.close()
        tmp.unlink(missing_ok=True)


def _write_silent_wav(path: Path, duration_sec: float = 2., rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x00\x00" * int(duration_sec * rate))
    return path


def transcribe_wav(wav_path: str | Path, language: str = "pt-BR", provider: str | None = None) -> dict:
    provider = provider or os.getenv("SPEECH_PROVIDER", "whisper")
    if provider == "whisper":
        from src.audio.whisper_transcriber import transcribe_whisper
        return transcribe_whisper(wav_path, language)
    if provider == "azure":
        from src.cloud.azure_services import transcribe_azure
        return transcribe_azure(wav_path, language)
    result = {"engine": provider, "status": "unavailable", "transcript": "",
              "cloud_executed": False, "wav_path": str(wav_path)}
    if provider == "offline":
        return result
    if provider != "legacy":
        raise ValueError("Provedor de áudio inválido")
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 15
        with sr.AudioFile(str(wav_path)) as source:
            text = recognizer.recognize_google(recognizer.record(source), language=language)
        return {**result, "engine": "speechrecognition-google", "status": "ok", "transcript": text,
                "cloud_executed": True}
    except Exception as exc:
        return {**result, "status": "error", "error_type": type(exc).__name__}


def transcribe_media(media_path: str | Path, work_dir: str | Path) -> dict:
    target = Path(work_dir) / f"{Path(media_path).stem}.wav"
    extract_audio_wav(media_path, target)
    return {**transcribe_wav(target), "source": str(media_path)}
