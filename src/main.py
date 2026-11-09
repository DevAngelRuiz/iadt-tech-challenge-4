#!/usr/bin/env python3
"""CLI principal — pipeline multimodal Saúde da Mulher (Tech Challenge Fase 4)."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.voice_risk import analyze_audio_risk
from src.config import OUTPUT_DIR
from src.fusion.multimodal_engine import fuse_modalities
from src.reporting.report_builder import save_run_outputs
from src.video.pipeline import run_video_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Análise multimodal (vídeo YOLOv8 + áudio + texto) para saúde da mulher"
    )
    p.add_argument("--video", type=str, help="Caminho do vídeo clínico")
    p.add_argument("--audio", type=str, help="Caminho de áudio de consulta (wav/mp3/mp4)")
    p.add_argument("--text", type=str, help="Texto clínico / transcrição manual")
    p.add_argument("--text-file", type=str, help="Arquivo de texto clínico")
    p.add_argument("--transcript-override", type=str, help="Força transcrição (demo sem STT real)")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--skip-emotion", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    p.add_argument("--prefix", type=str, default="run")
    p.add_argument("--speech-provider", choices=["whisper", "azure", "offline", "legacy"], default=os.getenv("SPEECH_PROVIDER", "whisper"))
    p.add_argument("--speech-language", default="pt-BR")
    p.add_argument("--nlp-provider", choices=["azure", "local", "aws", "qwen"], default=os.getenv("NLP_PROVIDER", "local"))
    p.add_argument("--require-qwen", action="store_true", help="Exige resposta real e validada do Qwen local")
    p.add_argument("--require-transcription", action="store_true", help="Exige fala transcrita automaticamente, sem override")
    p.add_argument("--require-azure", action="store_true", help="Falha se os dois serviços Azure não forem comprovados")
    p.add_argument("--vitals", help="CSV/JSONL de sinais vitais")
    p.add_argument("--prescriptions", help="CSV/JSONL de prescrições")
    p.add_argument("--movement", help="CSV/JSONL de atividade temporal")
    p.add_argument("--rules", help="JSON de regras para anomalias")
    p.add_argument("--alert-db", default="outputs/alerts.sqlite")
    p.add_argument("--monitor-interval", type=float, default=0.)
    p.add_argument("--max-video-frames", type=int, default=None, help="Limite explícito; padrão analisa o vídeo inteiro")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.monitor_interval < 0 or (args.max_video_frames is not None and args.max_video_frames <= 0):
        print("Intervalo/limite inválido", file=sys.stderr)
        return 2
    if args.require_transcription and (args.transcript_override is not None or not (args.audio or args.video)):
        print("Transcrição automática exige áudio/vídeo com fala, sem override.", file=sys.stderr)
        return 3
    if args.require_qwen and args.nlp_provider != "qwen":
        print("Modo Qwen exige --nlp-provider qwen.", file=sys.stderr)
        return 3
    if args.require_azure:
        from src.cloud.azure_services import azure_readiness
        if (not all(azure_readiness().values()) or args.speech_provider != "azure" or
                args.nlp_provider != "azure" or args.transcript_override is not None or not args.audio):
            print("Modo Azure exige os dois recursos configurados e --audio com fala, sem override.", file=sys.stderr)
            return 3
    if not any([args.video, args.audio, args.text, args.text_file, args.vitals, args.prescriptions, args.movement]):
        print("Informe ao menos --video, --audio, --text ou --text-file", file=sys.stderr)
        return 2

    text_input = args.text or ""
    if args.text_file:
        text_input = Path(args.text_file).read_text(encoding="utf-8")

    video_result = None
    from src.reporting.alert_store import AlertStore
    store, run_id = AlertStore(args.alert_db), uuid.uuid4().hex
    anomaly_result = None
    if any([args.vitals, args.prescriptions, args.movement]):
        from src.anomalies.engine import instant, load_events
        from src.monitor import run_events
        records = []
        try:
            for path, kind in [(args.vitals, "vitals"), (args.prescriptions, "prescription"), (args.movement, "movement")]:
                if path:
                    records.extend(load_events(path, kind))
            records.sort(key=lambda record: instant(record["timestamp"]))
            rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if args.rules else None
            anomaly_result = run_events(records, store, rules, args.monitor_interval, run_id)
        except (ValueError, KeyError, OSError) as exc:
            print(f"Dados de monitoramento inválidos: {exc}", file=sys.stderr)
            return 2
    if args.video:
        print(f"[video] Processando {args.video} ...")
        video_result = run_video_pipeline(
            args.video,
            output_dir=output_dir,
            enable_emotion=not args.skip_emotion,
            enable_pose=not args.skip_pose,
            max_frames=args.max_video_frames,
        )
        print(
            f"[video] bleeding={video_result['scores']['bleeding']} "
            f"distress={video_result['scores']['emotion_distress']} "
            f"pose={video_result['scores']['pose_closed']}"
        )

    audio_result = None
    if args.audio or args.transcript_override:
        audio_source = args.audio or args.video
        if not audio_source and args.transcript_override:
            # Cria WAV silencioso só para prosódia quando há override sem arquivo
            silent = output_dir / "audio_work" / f"silent_override_{run_id}.wav"
            silent.parent.mkdir(parents=True, exist_ok=True)
            from src.audio.transcriber import _write_silent_wav

            _write_silent_wav(silent, duration_sec=2.0)
            audio_source = str(silent)
        if audio_source:
            print(f"[audio] Processando {audio_source} ...")
            audio_result = analyze_audio_risk(
                audio_source,
                work_dir=output_dir / "audio_work",
                transcript_override=args.transcript_override,
                speech_provider=args.speech_provider,
                language=args.speech_language,
            )
            print(
                f"[audio] risk={audio_result['audio_risk_score']} "
                f"top={audio_result['top_label']} engine={audio_result['stt_engine']}"
            )
    elif args.video:
        print(f"[audio] Extraindo áudio de {args.video} ...")
        audio_result = analyze_audio_risk(
            args.video,
            work_dir=output_dir / "audio_work",
            transcript_override=None,
            speech_provider=args.speech_provider,
            language=args.speech_language,
        )
        print(
            f"[audio] risk={audio_result['audio_risk_score']} "
            f"top={audio_result['top_label']} engine={audio_result['stt_engine']}"
        )

    print("[fusion] Combinando modalidades ...")
    fusion = fuse_modalities(video_result=video_result, audio_result=audio_result, text_input=text_input,
                             nlp_provider=args.nlp_provider,
                             anomaly_alerts=anomaly_result["alerts"] if anomaly_result else None)
    if anomaly_result and anomaly_result["data_quality"]:
        fusion["data_quality"].extend(json.dumps(item, ensure_ascii=False) for item in anomaly_result["data_quality"])
        fusion["assessment_status"] = "partial"
    for alert in fusion["alerts"]:
        if "id" not in alert:
            store.publish(alert, run_id)
    print(f"[fusion] nível={fusion['alert_level']} final={fusion['scores']['final']}")

    payload = {
        "video": video_result,
        "audio": audio_result,
        "text_input": text_input,
        "fusion": fusion,
        "anomalies": anomaly_result,
        "run_id": run_id,
        "alert_db": args.alert_db,
    }
    paths = save_run_outputs(output_dir, payload, prefix=args.prefix)
    print(f"[ok] JSON: {paths['json']}")
    print(f"[ok] MD:   {paths['markdown']}")
    print(json.dumps({"alert_level": fusion["alert_level"], "scores": fusion["scores"], "alerts": fusion["alerts"]}, ensure_ascii=False, indent=2))
    if args.require_azure and ((audio_result or {}).get("status") != "ok" or fusion["cloud_nlp"].get("status") != "ok"):
        print("Azure não comprovado nesta execução; consulte o estado dos provedores no JSON.", file=sys.stderr)
        return 3
    if args.require_transcription and ((audio_result or {}).get("status") != "ok" or not (audio_result or {}).get("transcript", "").strip()):
        print("Transcrição automática não comprovada; consulte status/engine no JSON.", file=sys.stderr)
        return 3
    if args.require_qwen and (fusion["cloud_nlp"].get("status") != "ok" or not fusion["cloud_nlp"].get("model_inference_executed")):
        print("Qwen não comprovado; consulte status/engine no JSON.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
