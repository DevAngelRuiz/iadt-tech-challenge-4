"""Detector YOLOv8 customizado para sangramento anômalo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import YOLO_WEIGHTS


class BleedingDetector:
    """Detecta regiões de sangramento anômalo com YOLOv8."""

    def __init__(self, weights: Path | None = None, conf: float = 0.15):
        self.weights = Path(weights) if weights else YOLO_WEIGHTS
        self.conf = conf
        self.model = None
        self._hsv_fallback = True
        if self.weights.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.weights))
                self._hsv_fallback = False
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Falha ao carregar YOLO ({exc}); usando fallback HSV.")

    def detect_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.model is not None:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            detections: list[dict[str, Any]] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "label": "bleeding",
                            "confidence": float(box.conf[0]),
                            "bbox": [int(v) for v in xyxy],
                        }
                    )
            return detections
        return self._detect_hsv(frame)

    def _detect_hsv(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Fallback heurístico: máscaras vermelhas (demo sem pesos)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.005
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            detections.append(
                {
                    "label": "bleeding",
                    "confidence": min(0.95, 0.4 + area / (h * w)),
                    "bbox": [x, y, x + bw, y + bh],
                    "method": "hsv_fallback",
                }
            )
        return detections

    def annotate(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return out


def analyze_video(
    video_path: str | Path,
    output_video: str | Path | None = None,
    sample_every: int = 5,
    max_frames: int | None = 300,
) -> dict[str, Any]:
    """Processa vídeo com YOLOv8 (ou fallback) e retorna eventos agregados."""
    detector = BleedingDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    events: list[dict[str, Any]] = []
    frame_idx = 0
    frames_with_bleeding = 0
    max_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            dets = detector.detect_frame(frame)
            if dets:
                frames_with_bleeding += 1
                max_conf = max(max_conf, max(d["confidence"] for d in dets))
                events.append({"frame": frame_idx, "t_sec": frame_idx / fps, "detections": dets})
            if writer is not None:
                frame = detector.annotate(frame, dets)
        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    sampled = max(1, (frame_idx + sample_every - 1) // sample_every)
    ratio = frames_with_bleeding / sampled
    # Prioriza presença + confiança máxima
    score = min(1.0, ratio * 0.85 + max_conf * 0.55)
    if frames_with_bleeding > 0:
        score = max(score, min(1.0, 0.35 + max_conf * 0.5))
    return {
        "modality": "video_yolo",
        "model": "YOLOv8-bleeding" if detector.model else "HSV-fallback",
        "frames_processed": frame_idx,
        "total_frames": total_frames,
        "coverage": round(frame_idx / total_frames, 4) if total_frames > 0 else None,
        "frames_with_bleeding": frames_with_bleeding,
        "max_confidence": max_conf,
        "bleeding_score": round(score, 3),
        "events": events[:50],
        "output_video": str(output_video) if output_video else None,
    }
