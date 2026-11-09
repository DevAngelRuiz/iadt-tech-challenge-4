"""Detecção de pose / postura com MediaPipe (aula FIAP 03) + fallback OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _closed_posture(landmarks) -> bool:
    """Postura fechada: braços cruzados / ombros contraídos (proxy de desconforto)."""
    ls, rs = landmarks[11], landmarks[12]
    lw, rw = landmarks[15], landmarks[16]
    shoulder_width = abs(rs.x - ls.x) + 1e-6
    wrists_close = abs(lw.x - rw.x) < shoulder_width * 0.6
    wrists_inward = (lw.x > ls.x) and (rw.x < rs.x)
    return wrists_close or wrists_inward


def _arm_up(landmarks) -> bool:
    le, re = landmarks[13], landmarks[14]
    ls, rs = landmarks[11], landmarks[12]
    return (le.y < ls.y) or (re.y < rs.y)


def _pose_visible(landmarks) -> bool:
    return all(
        np.isfinite([landmarks[index].x, landmarks[index].y, landmarks[index].visibility]).all()
        and landmarks[index].visibility >= .5
        for index in (11, 12, 13, 14, 15, 16)
    )


def _analyze_mediapipe_solutions(video_path: Path, sample_every: int, max_frames: int | None) -> dict[str, Any] | None:
    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            return None
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    except Exception:  # noqa: BLE001
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose.close()
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    closed_count = 0
    visible_samples = detected_samples = 0
    samples = 0
    arm_up = False
    arm_movements = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                detected_samples += 1
            if results.pose_landmarks and _pose_visible(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark
                visible_samples += 1
                if _closed_posture(lm):
                    closed_count += 1
                if _arm_up(lm):
                    if not arm_up:
                        arm_up = True
                        arm_movements += 1
                else:
                    arm_up = False
            else:
                arm_up = False
            samples += 1
        frame_idx += 1

    cap.release()
    pose.close()
    closed_score = (closed_count / visible_samples) if visible_samples else 0.0
    return {
        "modality": "video_pose",
        "model": "MediaPipe Pose",
        "status": "ok" if visible_samples else "unavailable",
        "reason": None if visible_samples else "no_visible_pose",
        "pose_available": bool(visible_samples),
        "person_detected": bool(detected_samples),
        "frames_with_visible_pose": visible_samples,
        "pose_coverage": round(visible_samples / samples, 3) if samples else 0.,
        "frames_sampled": samples,
        "closed_frames": closed_count,
        "pose_closed_score": round(closed_score, 3),
        "arm_movements": arm_movements,
        "physio_activity_score": round(min(1.0, arm_movements / 10.0), 3),
    }


def _analyze_motion_fallback(video_path: Path, sample_every: int, max_frames: int | None) -> dict[str, Any]:
    """Mede diferenças entre frames; não identifica pessoa, braços ou postura."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    prev = None
    motion_vals: list[float] = []
    frame_idx = 0
    samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                motion_vals.append(float(np.mean(diff)) / 255.0)
            prev = gray
            samples += 1
        frame_idx += 1
    cap.release()

    mean_motion = float(np.mean(motion_vals)) if motion_vals else 0.0

    return {
        "modality": "video_pose",
        "model": "OpenCV-motion-fallback",
        "status": "unavailable",
        "reason": "pose_model_unavailable",
        "pose_available": False,
        "person_detected": None,
        "activity_method": "frame_difference_without_person_detection",
        "frames_sampled": samples,
        "closed_frames": 0,
        "pose_closed_score": 0.,
        "arm_movements": None,
        "physio_activity_score": round(min(1.0, mean_motion * 10), 3),
        "mean_motion": round(mean_motion, 4),
    }


def analyze_video_pose(
    video_path: str | Path,
    sample_every: int = 5,
    max_frames: int | None = 240,
) -> dict[str, Any]:
    path = Path(video_path)
    result = _analyze_mediapipe_solutions(path, sample_every, max_frames)
    if result is not None:
        return result
    return _analyze_motion_fallback(path, sample_every, max_frames)
