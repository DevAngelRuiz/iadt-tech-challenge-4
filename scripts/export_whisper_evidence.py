"""Preserva evidências textuais da inferência local; não executa o modelo."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text_analysis_metadata(payload):
    nlp = payload.get("fusion", {}).get("cloud_nlp") or {}
    classifier = payload.get("fusion", {}).get("text_classification") or {}
    return {"engine": nlp.get("engine", "unknown"), "status": nlp.get("status", "unknown"),
            "model": nlp.get("model"), "cloud_executed": nlp.get("cloud_executed", False),
            "model_inference_executed": nlp.get("model_inference_executed", False),
            "risk_classifier": classifier.get("model")}


def main():
    source = ROOT / "outputs" / "whisper_validation"
    speech = max(source.glob("whisper_*.json"), key=lambda p: p.stat().st_mtime_ns)
    tone = max((source / "tone").glob("tone_*.json"), key=lambda p: p.stat().st_mtime_ns)
    payload = json.loads(speech.read_text(encoding="utf-8"))
    control = json.loads(tone.read_text(encoding="utf-8"))
    audio = payload["audio"]
    assert audio["stt_engine"] == "whisper-local" and audio["status"] == "ok"
    assert audio["model_inference_executed"] and not audio["cloud_executed"]
    assert not control["fusion"]["alerts"] and control["audio"]["status"] == "no_speech"
    reference = (ROOT / "data/samples/consulta_falada_sintetica.txt").read_text(encoding="utf-8-sig").strip()
    target = ROOT / "outputs/examples/whisper"
    target.mkdir(parents=True, exist_ok=True)
    for path, name in [(speech, "fala"), (tone, "controle_tons")]:
        (target / (name + ".json")).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        (target / (name + ".md")).write_text(path.with_suffix(".md").read_text(encoding="utf-8").replace(path.name, name + ".json"), encoding="utf-8")
    summary = {"provider": audio["stt_engine"], "model": audio["stt_model"], "device": audio["stt_device"],
               "actual_inference": audio["model_inference_executed"], "cloud_executed": False,
               "synthetic_speech": True, "voice": "Microsoft Maria Desktop", "language": "pt-BR",
               "reference": reference, "transcript": audio["transcript"],
               "exact_reference_match_this_clip": audio["transcript"].strip() == reference,
               "tone_status": control["audio"]["status"], "tone_alerts": len(control["fusion"]["alerts"]),
               "text_analysis": text_analysis_metadata(payload),
               "scope": "One synthetic clip and one tone control; not a general accuracy benchmark"}
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
