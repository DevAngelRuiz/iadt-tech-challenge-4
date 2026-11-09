"""Verifica configuração sem revelar segredos; --audio executa a prova real dos dois serviços."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # carrega o .env antes de verificar as variáveis
from src.cloud.azure_services import azure_readiness


def main():
    config.load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio")
    parser.add_argument("--output-dir", default="outputs/azure_validation")
    args = parser.parse_args()
    readiness = azure_readiness()
    print(json.dumps({"configured": readiness, "live_validation": "not_run"}, ensure_ascii=False, indent=2))
    if not args.audio:
        print("Para comprovar: acrescente --audio com um arquivo de fala. Consulte GUIA_AZURE.md.")
        return 0 if all(readiness.values()) else 3
    from src.main import main as pipeline
    return pipeline(["--audio", args.audio, "--speech-provider", "azure", "--nlp-provider", "azure",
                     "--require-azure", "--prefix", "azure", "--output-dir", args.output_dir])


if __name__ == "__main__":
    raise SystemExit(main())
