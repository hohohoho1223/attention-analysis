from dataclasses import dataclass
import math

@dataclass
class UpperBodyState:
    shoulder_tilt: float = 0.0
    body_visible: bool = False


class UpperBodyAnalyzer:
    """
    Upper body analyzer using MediaPipe Pose landmarks.
    Uses shoulder landmarks (11, 12) to estimate body tilt.
    """

    LEFT_SHOULDER = 11 # MediaPipe Pose에서 왼쪽 어깨 랜드마크 인덱스
    RIGHT_SHOULDER = 12 # MediaPipe Pose에서 오른쪽 어깨 랜드마크 인덱스

    def estimate(self, pose_landmarks, frame_width: int, frame_height: int) -> UpperBodyState:
        if pose_landmarks is None:
            return UpperBodyState(shoulder_tilt=0.0, body_visible=False)

        l = pose_landmarks.landmark[self.LEFT_SHOULDER]
        r = pose_landmarks.landmark[self.RIGHT_SHOULDER]

        x1, y1 = l.x * frame_width, l.y * frame_height
        x2, y2 = r.x * frame_width, r.y * frame_height

        dx = x2 - x1
        dy = y2 - y1

        angle = math.degrees(math.atan2(dy, dx))

        return UpperBodyState(
            shoulder_tilt=angle,
            body_visible=True,
        )