#!/usr/bin/env python3
"""Treina YOLOv8n customizado para classe `bleeding`."""

from __future__ import annotations

import shutil
import sys
import json
import hashlib
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, YOLO_DATA_YAML  # noqa: E402


def main() -> int:
    if not YOLO_DATA_YAML.exists():
        print("Dataset ausente. Rode: python scripts/prepare_sample_data.py")
        return 1

    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = yaml.safe_load(YOLO_DATA_YAML.read_text(encoding="utf-8"))
    dataset["path"] = str(YOLO_DATA_YAML.parent.resolve())
    runtime_yaml = MODELS_DIR / "bleeding_dataset.runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(runtime_yaml),
        epochs=15,
        imgsz=640,
        batch=8,
        project=str(ROOT / "runs"),
        name="bleeding_yolov8n",
        exist_ok=True,
        verbose=True,
    )
    # Copia best.pt para models/
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    dest = MODELS_DIR / "bleeding_yolov8n.pt"
    if best.exists():
        shutil.copy2(best, dest)
        print(f"[ok] Pesos salvos em {dest}")
    else:
        print("[warn] best.pt não encontrado; verifique runs/")
        return 1

    # Validação rápida
    metrics = YOLO(str(dest)).val(data=str(runtime_yaml))
    exported = {"model": "YOLOv8n-bleeding", "epochs": 15,
                "dataset": "data/yolo_bleeding (synthetic)", "weights": str(dest.relative_to(ROOT)),
                "weights_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "metrics": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)},
                "note": "Exportado da validação do peso best.pt entregue; validação sintética."}
    (MODELS_DIR / "bleeding_metrics.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print("[ok] Validação:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
