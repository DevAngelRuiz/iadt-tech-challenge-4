"""Inicia llama.cpp em primeiro plano, somente em loopback, com contexto reduzido."""
import argparse
from pathlib import Path
import subprocess
from setup_qwen import ROOT, MODEL, MODEL_SHA, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "models" / "qwen")
    args = p.parse_args()
    model = args.directory / MODEL
    servers = list(args.directory.glob("runtime/**/llama-server.exe"))
    if not model.exists() or not servers:
        print("Qwen não instalado. Consulte GUIA_QWEN.md e scripts/setup_qwen.py.")
        return 4
    if digest(model) != MODEL_SHA:
        print("SHA256 do modelo inválido; execução não iniciada.")
        return 4
    return subprocess.call([str(servers[0].resolve()), "-m", str(model.resolve()),
                            "--alias", "qwen2.5-1.5b-instruct", "--host", "127.0.0.1",
                            "--port", "8081", "-c", "4096", "-np", "1", "-t", "2", "-ngl", "0"])


if __name__ == "__main__":
    raise SystemExit(main())
