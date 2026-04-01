from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [473, 474, 475, 476]
RIGHT_IRIS_INDICES = [468, 469, 470, 471]


@dataclass
class EyeFocusResult:
    gaze_direction: str = "Unknown"
    blink_bpm: int = 0
    eye_focus_score: float = 100.0
    eye_status_msg: str = "Eye analysis disabled"
    is_calibrated: bool = False
    calibration_progress: int = 0
    smooth_ear: float = 0.0
    smooth_gaze_ratio: float = 0.5


class EyeFocusAnalyzer:
    def __init__(self) -> None:
        self.left_eye_indices = LEFT_EYE_INDICES
        self.right_eye_indices = RIGHT_EYE_INDICES
        self.left_iris_indices = LEFT_IRIS_INDICES
        self.right_iris_indices = RIGHT_IRIS_INDICES

        self.ear_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=15)
        self.gaze_variance_buffer = deque(maxlen=30)
        self.blink_timestamps = deque()

        self.is_calibrated = False
        self.calibration_frames = 0
        self.max_calibration_frames = 100
        self.base_ear_sum = 0.0
        self.base_ear = 0.0

        self.eye_closed = False
        self.start_time = time.time()
        self.last_result = EyeFocusResult()

    def reset(self) -> None:
        self.gaze_buffer.clear()
        self.gaze_variance_buffer.clear()
        self.blink_timestamps.clear()
        self.eye_closed = False
        self.last_result = EyeFocusResult(
            gaze_direction="Unknown",
            blink_bpm=0,
            eye_focus_score=0.0,
            eye_status_msg="Face Not Detected (Absence)",
            is_calibrated=self.is_calibrated,
            calibration_progress=self.calibration_frames,
        )

    def _euclidean_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _get_coords(
        self,
        face_landmarks,
        indices: list[int],
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        return [
            (
                int(face_landmarks.landmark[i].x * width),
                int(face_landmarks.landmark[i].y * height),
            )
            for i in indices
        ]

    def _calculate_ear(self, eye_points: list[tuple[int, int]]) -> float | None:
        try:
            v1 = self._euclidean_distance(eye_points[1], eye_points[5])
            v2 = self._euclidean_distance(eye_points[2], eye_points[4])
            h = self._euclidean_distance(eye_points[0], eye_points[3])
            if h <= 0:
                return None
            return (v1 + v2) / (2.0 * h)
        except Exception:
            return None

    def _get_gaze_ratio(
        self,
        eye_points: list[tuple[int, int]],
        iris_points: list[tuple[int, int]],
    ) -> tuple[float, float]:
        try:
            eye_corners_x = sorted([eye_points[0][0], eye_points[3][0]])
            eye_x_min, eye_x_max = eye_corners_x[0], eye_corners_x[1]
            iris_center_x = float(np.mean(iris_points, axis=0)[0])

            total_width = eye_x_max - eye_x_min
            if total_width <= 0:
                return 0.5, 0.0
            
            # 눈 안에서 홍채가 어디 위치해 있는지
            ratio = (iris_center_x - eye_x_min) / total_width
            return ratio, float(total_width)
        except Exception:
            return 0.5, 0.0

    def _is_face_clipped(self, face_landmarks) -> bool:
        x_coords = [lm.x for lm in face_landmarks.landmark]
        y_coords = [lm.y for lm in face_landmarks.landmark]
        return (
            min(x_coords) < 0.01
            or max(x_coords) > 0.99
            or min(y_coords) < 0.01
            or max(y_coords) > 0.99
        )

    def _update_calibration(self, smooth_ear: float) -> EyeFocusResult:
        self.calibration_frames += 1
        self.base_ear_sum += smooth_ear
        progress = min(100, self.calibration_frames)

        if self.calibration_frames >= self.max_calibration_frames:
            self.base_ear = self.base_ear_sum / self.max_calibration_frames
            self.is_calibrated = True
            progress = 100

        result = EyeFocusResult(
            gaze_direction="Unknown",
            blink_bpm=0,
            eye_focus_score=100.0,
            eye_status_msg=f"Calibrating... {progress}%",
            is_calibrated=self.is_calibrated,
            calibration_progress=progress,
            smooth_ear=smooth_ear,
            smooth_gaze_ratio=0.5,
        )
        self.last_result = result
        return result

    def _update_blink_bpm(self, smooth_ear: float, current_time: float) -> int:
        blink_threshold = self.base_ear * 0.75

        if smooth_ear < blink_threshold:
            if not self.eye_closed:
                self.blink_timestamps.append(current_time)
                self.eye_closed = True
        else:
            self.eye_closed = False

        while self.blink_timestamps and current_time - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()

        return len(self.blink_timestamps)

    # 눈(홍채) 좌우 구분
    def _classify_gaze(self, smooth_gaze: float) -> str:
        if smooth_gaze < 0.44:
            return "Right"
        if smooth_gaze > 0.55:
            return "Left"
        return "Center"

    def _build_focus_result(
        self,
        gaze_direction: str,
        blink_bpm: int,
        smooth_ear: float,
        smooth_gaze: float,
        current_time: float,
    ) -> EyeFocusResult:
        self.gaze_variance_buffer.append(smooth_gaze)
        gaze_variance = np.var(self.gaze_variance_buffer) if len(self.gaze_variance_buffer) > 20 else 1.0
        elapsed_time = current_time - self.start_time

        eye_status_msg = "Focused (Optimal)"
        eye_focus_score = 100.0

        # 시선 이탈 감지: 중앙에서 벗어난 시선이 지속될 때
        if gaze_direction != "Center":
            eye_status_msg = "Distracted (Looking Away)"
            eye_focus_score = 40.0
        # 과도한 깜빡임 감지: 15 BPM 이상일 때
        elif blink_bpm > 15:
            eye_status_msg = "Anxious/Distracted (High BPM)"
            eye_focus_score = 60.0
        # 멍때림 감지: 10초 이상 고정된 시선과 낮은 깜빡임 BPM이 지속될 때
        elif elapsed_time > 10 and blink_bpm < 3 and gaze_variance < 0.0005:
            eye_status_msg = "Spacing Out (Low BPM & Fixed Gaze)"
            eye_focus_score = 50.0

        result = EyeFocusResult(
            gaze_direction=gaze_direction,
            blink_bpm=blink_bpm,
            eye_focus_score=eye_focus_score,
            eye_status_msg=eye_status_msg,
            is_calibrated=self.is_calibrated,
            calibration_progress=min(self.calibration_frames, 100),
            smooth_ear=smooth_ear,
            smooth_gaze_ratio=smooth_gaze,
        )
        self.last_result = result
        return result

    def analyze(self, frame, face_landmarks) -> EyeFocusResult:
        height, width = frame.shape[:2]

        if face_landmarks is None:
            self.reset()
            return self.last_result

        if self._is_face_clipped(face_landmarks):
            result = EyeFocusResult(
                gaze_direction=self.last_result.gaze_direction,
                blink_bpm=self.last_result.blink_bpm,
                eye_focus_score=self.last_result.eye_focus_score if self.is_calibrated else 80.0,
                eye_status_msg="Face Clipped (Too close to edge)",
                is_calibrated=self.is_calibrated,
                calibration_progress=min(self.calibration_frames, 100),
                smooth_ear=self.last_result.smooth_ear,
                smooth_gaze_ratio=self.last_result.smooth_gaze_ratio,
            )
            self.last_result = result
            return result

        left_eye_coords = self._get_coords(face_landmarks, self.left_eye_indices, width, height)
        right_eye_coords = self._get_coords(face_landmarks, self.right_eye_indices, width, height)
        left_iris_coords = self._get_coords(face_landmarks, self.left_iris_indices, width, height)
        right_iris_coords = self._get_coords(face_landmarks, self.right_iris_indices, width, height)

        ear_l = self._calculate_ear(left_eye_coords)
        ear_r = self._calculate_ear(right_eye_coords)

        if ear_l is None or ear_r is None:
            result = EyeFocusResult(
                gaze_direction="Unknown",
                blink_bpm=0,
                eye_focus_score=80.0,
                eye_status_msg="Eye landmarks unstable",
                is_calibrated=self.is_calibrated,
                calibration_progress=min(self.calibration_frames, 100),
            )
            self.last_result = result
            return result

        avg_ear = (ear_l + ear_r) / 2.0
        self.ear_buffer.append(avg_ear)
        smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)

        if not self.is_calibrated:
            return self._update_calibration(smooth_ear)

        current_time = time.time()
        blink_bpm = self._update_blink_bpm(smooth_ear, current_time)

        ratio_l, width_l = self._get_gaze_ratio(left_eye_coords, left_iris_coords)
        ratio_r, width_r = self._get_gaze_ratio(right_eye_coords, right_iris_coords)

        selected_eye = "LeftEye" if width_l > width_r else "RightEye"
        current_gaze_ratio = ratio_l if width_l > width_r else ratio_r

        # 시선 버퍼에 추가하고 평균 계산 → “최근 여러 프레임이 누적되어야 방향이 안정적으로 바뀜(노이즈 완화)”
        self.gaze_buffer.append(current_gaze_ratio)
        smooth_gaze = sum(self.gaze_buffer) / len(self.gaze_buffer)
        gaze_direction = self._classify_gaze(smooth_gaze)

        print(
            "[GAZE DEBUG] "
            f"ratio_l={ratio_l:.3f} (w={width_l:.1f}) | "
            f"ratio_r={ratio_r:.3f} (w={width_r:.1f}) | "
            f"selected={selected_eye} | "
            f"current={current_gaze_ratio:.3f} | "
            f"smooth={smooth_gaze:.3f} | "
            f"direction={gaze_direction}"
        )

        return self._build_focus_result(
            gaze_direction=gaze_direction,
            blink_bpm=blink_bpm,
            smooth_ear=smooth_ear,
            smooth_gaze=smooth_gaze,
            current_time=current_time,
        )