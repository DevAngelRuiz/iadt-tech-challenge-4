"""Whisper local: recebe PCM já normalizado, sem exigir FFmpeg no PATH."""
from functools import lru_cache
import os
from pathlib import Path
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_model(name, device, cache_dir):
    import whisper
    return whisper.load_model(name, device=device, download_root=cache_dir)


def transcribe_whisper(wav_path, language="pt-BR"):
    name = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    cache = Path(os.getenv("WHISPER_CACHE_DIR", str(ROOT / "models" / "whisper")))
    if not cache.is_absolute():
        cache = ROOT / cache
    result = {"engine": "whisper-local", "status": "unavailable", "transcript": "",
              "cloud_executed": False, "model_inference_executed": False,
              "model": name, "device": device, "language": language.split("-")[0].lower(),
              "segments": [], "wav_path": str(wav_path)}
    try:
        with wave.open(str(wav_path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
                return {**result, "status": "invalid_audio"}
            audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    except (OSError, ValueError, wave.Error, EOFError) as exc:
        return {**result, "status": "invalid_audio", "error_type": type(exc).__name__}
    if not audio.size or np.max(np.abs(audio)) < 1e-5:
        return {**result, "status": "no_speech", "reason": "silence"}
    try:
        import whisper  # noqa: F401 — dependência opcional identificada antes do download
    except ImportError:
        return {**result, "status": "dependency_missing"}
    try:
        model = _load_model(name, device, str(cache))
        decoded = model.transcribe(audio, language=result["language"], task="transcribe", fp16=False,
                                   temperature=0, condition_on_previous_text=False, verbose=None,
                                   no_speech_threshold=.6, logprob_threshold=-1., compression_ratio_threshold=2.4)
        result["model_inference_executed"] = True
        for segment in decoded.get("segments", []):
            if (segment.get("no_speech_prob", 0) > .6 or segment.get("avg_logprob", 0) < -1.
                    or segment.get("compression_ratio", 0) > 2.4):
                continue
            if segment.get("text", "").strip():
                result["segments"].append({"text": segment["text"].strip(),
                                           "start_seconds": segment["start"], "end_seconds": segment["end"],
                                           "no_speech_prob": segment.get("no_speech_prob"),
                                           "avg_logprob": segment.get("avg_logprob")})
        result["transcript"] = " ".join(segment["text"] for segment in result["segments"]).strip()
        result["status"] = "ok" if result["transcript"] else "no_speech"
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__)
    return result
