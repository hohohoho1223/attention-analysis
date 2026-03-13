from __future__ import annotations

import math
from dataclasses import dataclass, field


UPPER_BODY_LANDMARKS = {
    "left_shoulder": 11, # MediaPipe Pose에서 왼쪽 어깨 랜드마크 인덱스
    "right_shoulder": 12, # MediaPipe Pose에서 오른쪽 어깨 랜드마크 인덱스
    "left_elbow": 13, # MediaPipe Pose에서 왼쪽 팔꿈치 랜드마크 인덱스
    "right_elbow": 14, # MediaPipe Pose에서 오른쪽 팔꿈치 랜드마크 인덱스
}


@dataclass
class UpperBodyState:
    shoulder_tilt: float = 0.0
    body_visible: bool = False
    landmark_points: dict[str, tuple[int, int]] = field(default_factory=dict)


class UpperBodyAnalyzer:
    """
    Upper body analyzer using MediaPipe Pose landmarks.
    Uses shoulder landmarks (11, 12) to estimate body tilt.
    """

    def _extract_landmark_points(
        self,
        pose_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, tuple[int, int]]:
        points: dict[str, tuple[int, int]] = {}

        for name, idx in UPPER_BODY_LANDMARKS.items():
            landmark = pose_landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            points[name] = (x, y)

        return points

    def _calculate_shoulder_tilt(
        self,
        points: dict[str, tuple[int, int]],
    ) -> float:
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]

        x1, y1 = left_shoulder
        x2, y2 = right_shoulder

        dx = x2 - x1
        dy = y2 - y1

        return math.degrees(math.atan2(dy, dx))

    def estimate(
        self,
        pose_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> UpperBodyState:
        if pose_landmarks is None:
            return UpperBodyState()

        points = self._extract_landmark_points(
            pose_landmarks,
            frame_width,
            frame_height,
        )
        shoulder_tilt = self._calculate_shoulder_tilt(points)

        return UpperBodyState(
            shoulder_tilt=shoulder_tilt,
            body_visible=True,
            landmark_points=points,
        )