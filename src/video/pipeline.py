"""Pipeline de vídeo multimodal."""

from __future__ import annotations

from pathlib import Path
import uuid
from typing import Any

from src.video.emotion_analyzer import analyze_video_emotions
from src.video.pose_analyzer import analyze_video_pose
from src.video.yolo_detector import analyze_video


def run_video_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    enable_emotion: bool = True,
    enable_pose: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = output_dir / f"{Path(video_path).stem}_{uuid.uuid4().hex}_yolo_annotated.mp4"

    yolo = analyze_video(video_path, output_video=annotated, max_frames=max_frames)
    emotion = analyze_video_emotions(video_path, max_frames=max_frames) if enable_emotion else {"distress_score": 0.0}
    pose = analyze_video_pose(video_path, max_frames=max_frames) if enable_pose else {"pose_closed_score": 0.0}

    return {
        "video_path": str(video_path),
        "yolo": yolo,
        "emotion": emotion,
        "pose": pose,
        "scores": {
            "bleeding": yolo.get("bleeding_score", 0.0),
            "emotion_distress": emotion.get("distress_score", 0.0),
            "pose_closed": pose.get("pose_closed_score", 0.0),
        },
    }
