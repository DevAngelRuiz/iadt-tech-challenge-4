"""Provisionamento opcional, fixado por revisão e SHA256. Sem download por padrão."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
MODEL_URL = f"https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/{REVISION}/{MODEL}"
MODEL_SHA = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
MODEL_SIZE = 1117320736
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12188/llama-b12188-bin-win-cpu-x64.zip"
RUNTIME_SHA = "5d2b03ad405732693ebc93e0aa243c22ead5bf9c973e90cd911fb7f760b9d136"
RUNTIME_SIZE = 24749454
RESERVE = 2 * 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, destination, expected, size):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"SHA256 diferente: {destination}. Arquivo preservado para inspeção.")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise ValueError(f"Download parcial existente: {partial}. Verifique antes de repetir.")
    h, count = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, partial.open("xb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            count += len(chunk)
            if count > size:
                raise ValueError("Tamanho excedeu os metadados fixados")
            output.write(chunk)
            h.update(chunk)
    if count != size or h.hexdigest() != expected:
        raise ValueError(f"Tamanho/SHA256 inválido: {partial}")
    partial.rename(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models" / "qwen")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    target = args.directory.resolve()
    parent = target
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    model = target / MODEL
    archive = target / "llama-cpu-b12188.zip"
    needed = (0 if model.exists() else MODEL_SIZE) + (0 if archive.exists() else RUNTIME_SIZE) + 200 * 1024**2
    ready = model.exists() and any(target.glob("runtime/**/llama-server.exe"))
    status = {"free_bytes": free, "estimated_additional_bytes": needed,
              "reserve_bytes": RESERVE, "space_ok": free >= needed + RESERVE,
              "files_present": ready, "inference_validated": False,
              "directory": str(target), "model_revision": REVISION, "runtime": "b12188"}
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not args.download:
        return 0
    if not status["space_ok"]:
        print("Download não iniciado: libere espaço para manter 2 GiB de reserva, ou escolha outro disco.")
        return 4
    target.mkdir(parents=True, exist_ok=True)
    download(MODEL_URL, model, MODEL_SHA, MODEL_SIZE)
    download(RUNTIME_URL, archive, RUNTIME_SHA, RUNTIME_SIZE)
    runtime = target / "runtime"
    runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if not (runtime.resolve() / member.filename).resolve().is_relative_to(runtime.resolve()):
                raise ValueError("Caminho inválido no pacote")
            if member.file_size > 200 * 1024**2:
                raise ValueError("Membro maior que o orçamento de extração")
        if sum(m.file_size for m in package.infolist()) > 200 * 1024**2:
            raise ValueError("Pacote excede o orçamento de extração")
        package.extractall(runtime)
    (target / "provenance.json").write_text(json.dumps({
        "model_url": MODEL_URL, "model_sha256": MODEL_SHA,
        "runtime_url": RUNTIME_URL, "runtime_sha256": RUNTIME_SHA,
        "inference_validated": False}, indent=2), encoding="utf-8")
    print("Arquivos preparados. Inicie scripts/run_qwen_server.py e valide a inferência.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
