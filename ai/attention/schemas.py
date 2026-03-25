from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from attention.analyzers.attention_logic import AttentionState
from attention.analyzers.head_pose import PoseAngles
from attention.analyzers.upperbody_pose import UpperBodyState


@dataclass
class AttentionFrameResult:
    face_detected: bool
    pose_angles: PoseAngles
    gaze_direction: str
    blink_bpm: int
    eye_focus_score: float
    eye_status_msg: str
    upper_body_state: UpperBodyState
    attention_state: AttentionState
    is_drowsy: bool
    current_face_landmarks: Optional[Any] = None