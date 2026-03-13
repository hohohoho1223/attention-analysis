from __future__ import annotations

from dataclasses import dataclass

from head_pose import PoseAngles


@dataclass(frozen=True)
class AttentionConfig:
    yaw_threshold: float = 25.0
    pitch_threshold: float = 20.0
    roll_threshold: float = 15.0
    focused_yaw_threshold: float = 15.0
    focused_pitch_threshold: float = 10.0
    focused_roll_threshold: float = 10.0
    minor_distraction_time: float = 1.0
    distracted_time: float = 3.0
    lost_focus_time: float = 5.0
    no_face_time: float = 5.0
    smoothing_alpha: float = 0.35 # 스무딩 가중치 0.35(0.0 ~ 1.0)로 설정-> 현재 측정값이 35%, 이전 측정값 65% 반영
    recovery_speed: float = 1.5


DEFAULT_ATTENTION_CONFIG = AttentionConfig()


@dataclass
class AttentionState:
    state: str = "FOCUSED"
    score: float = 100.0
    distracted_duration: float = 0.0 # 집중 저하 지속 시간
    no_face_duration: float = 0.0 # 얼굴 미검출 지속 시간
    face_detected: bool = False
    smoothed_yaw: float = 0.0
    smoothed_pitch: float = 0.0
    smoothed_roll: float = 0.0


class AttentionAnalyzer:
    def __init__(
        self,
        config: AttentionConfig = DEFAULT_ATTENTION_CONFIG,
        state: AttentionState | None = None,
    ) -> None:
        self.config = config
        self.state = state if state is not None else AttentionState()

    def _smooth_value(self, previous: float, current: float) -> float:
        return self.config.smoothing_alpha * current + (1.0 - self.config.smoothing_alpha) * previous

    def _smooth_pose(self, pose_angles: PoseAngles) -> None:
        self.state.smoothed_yaw = self._smooth_value(self.state.smoothed_yaw, pose_angles.yaw)
        self.state.smoothed_pitch = self._smooth_value(self.state.smoothed_pitch, pose_angles.pitch)
        self.state.smoothed_roll = self._smooth_value(self.state.smoothed_roll, pose_angles.roll)

    def _is_focused_range(self) -> bool:
        return (
            abs(self.state.smoothed_yaw) <= self.config.focused_yaw_threshold
            and abs(self.state.smoothed_pitch) <= self.config.focused_pitch_threshold
            and abs(self.state.smoothed_roll) <= self.config.focused_roll_threshold
        )

    def _is_over_threshold(self) -> bool:
        return (
            abs(self.state.smoothed_yaw) > self.config.yaw_threshold
            or abs(self.state.smoothed_pitch) > self.config.pitch_threshold
            or abs(self.state.smoothed_roll) > self.config.roll_threshold
        )

    # 집중 저하 지속 시간 업데이트: 얼굴이 감지되고, 일정 각도 이상으로 고개가 돌아간 경우 지속 시간 증가, 그렇지 않고 집중 범위에 있으면 지속 시간 감소
    def _update_distracted_duration(self, dt: float) -> None:
        if self._is_over_threshold():
            self.state.distracted_duration += dt
            return

        if self._is_focused_range():
            self.state.distracted_duration = max(
                0.0,
                self.state.distracted_duration - dt * self.config.recovery_speed,
            )

    def _calculate_score(self) -> float:
        yaw_penalty = min(abs(self.state.smoothed_yaw) / max(self.config.yaw_threshold, 1e-6), 2.0) * 25.0
        pitch_penalty = min(abs(self.state.smoothed_pitch) / max(self.config.pitch_threshold, 1e-6), 2.0) * 20.0
        roll_penalty = min(abs(self.state.smoothed_roll) / max(self.config.roll_threshold, 1e-6), 2.0) * 10.0
        duration_penalty = min(self.state.distracted_duration, 5.0) * 8.0
        no_face_penalty = min(self.state.no_face_duration, 3.0) * 12.0

        score = 100.0 - yaw_penalty - pitch_penalty - roll_penalty - duration_penalty - no_face_penalty
        return max(0.0, min(100.0, score))

    def _update_state_label(self) -> None:
        if self.state.no_face_duration >= self.config.no_face_time:
            self.state.state = "LOST_FOCUS"
            return

        if self.state.distracted_duration >= self.config.lost_focus_time:
            self.state.state = "LOST_FOCUS"
            return

        if self.state.distracted_duration >= self.config.distracted_time:
            self.state.state = "DISTRACTED"
            return

        if self.state.distracted_duration >= self.config.minor_distraction_time:
            self.state.state = "MINOR_DISTRACTION"
            return

        self.state.state = "FOCUSED"

    # 업데이트 메서드: 얼굴 감지 여부와 각도 정보를 받아 상태를 업데이트하고, 최종적으로 현재 상태를 반환
    def update(self, pose_angles: PoseAngles, face_detected: bool, dt: float) -> AttentionState:
        self.state.face_detected = face_detected

        if face_detected:
            self.state.no_face_duration = 0.0
            self._smooth_pose(pose_angles)
            self._update_distracted_duration(dt)
        else:
            self.state.no_face_duration += dt
            self.state.distracted_duration += dt

        self.state.score = self._calculate_score()
        self._update_state_label()
        return self.state
