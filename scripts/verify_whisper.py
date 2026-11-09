"""Validação do Whisper local, sem Azure, API paga ou transcrição manual."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config


def main():
    config.load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio")
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--output-dir", default="outputs/whisper_validation")
    args = parser.parse_args()
    print(json.dumps({"package_installed": importlib.util.find_spec("whisper") is not None,
                      "model": os.getenv("WHISPER_MODEL", "base"), "device": os.getenv("WHISPER_DEVICE", "cpu"),
                      "api_key_required": False, "live_validation": "not_run"}, indent=2))
    if not args.audio:
        print("Acrescente --audio com uma gravação falada para comprovar a inferência local.")
        return 0
    from src.main import main as pipeline
    return pipeline(["--audio", args.audio, "--speech-provider", "whisper", "--speech-language", args.language,
                     "--nlp-provider", "local", "--require-transcription", "--output-dir", args.output_dir,
                     "--alert-db", str(Path(args.output_dir) / "alerts.sqlite"), "--prefix", "whisper"])


if __name__ == "__main__":
    raise SystemExit(main())
