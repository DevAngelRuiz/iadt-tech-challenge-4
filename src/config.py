"""Configuração central do projeto."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_environment(env_path: str | Path | None = None) -> bool:
    """Carrega o .env da raiz sem substituir variáveis já definidas no sistema."""
    return load_dotenv(dotenv_path=env_path if env_path is not None else ROOT / ".env", override=False)


load_environment()

DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
YOLO_DATA_DIR = DATA_DIR / "yolo_bleeding"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))

YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", MODELS_DIR / "bleeding_yolov8n.pt"))
YOLO_DATA_YAML = YOLO_DATA_DIR / "data.yaml"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Pesos da fusão multimodal
FUSION_WEIGHTS = {
    "bleeding": 0.35,
    "emotion_distress": 0.20,
    "pose_closed": 0.10,
    "audio_risk": 0.25,
    "text_risk": 0.10,
}

ALERT_THRESHOLDS = {
    "CRITICO": 0.70,
    "ATENCAO": 0.40,
}

EMOTION_DISTRESS = {"fear", "sad", "angry", "disgust"}

RISK_LEXICON = {
    "depressao_pos_parto": [
        "triste",
        "chorando",
        "não consigo dormir",
        "nao consigo dormir",
        "sem energia",
        "não quero viver",
        "nao quero viver",
        "culpa",
        "não me sinto mãe",
        "nao me sinto mae",
    ],
    "ansiedade": [
        "ansiosa",
        "ansiedade",
        "nervosa",
        "preocupada",
        "pânico",
        "panico",
        "coração acelerado",
        "coracao acelerado",
        "medo do parto",
    ],
    "violencia": [
        "ele me bate",
        "me machuca",
        "tenho medo dele",
        "não posso falar",
        "nao posso falar",
        "ameaça",
        "ameaca",
        "ele me controla",
        "controla meu telefone",
        "controla minhas mensagens",
        "me isola",
        "violencia",
        "violência",
    ],
    "fadiga": [
        "cansada",
        "exausta",
        "sem forças",
        "sem forcas",
        "fadiga",
        "não aguento",
        "nao aguento",
    ],
}
