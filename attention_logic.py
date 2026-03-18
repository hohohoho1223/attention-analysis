from __future__ import annotations

from dataclasses import dataclass

from head_pose import PoseAngles


@dataclass(frozen=True)
class AttentionConfig:
    yaw_threshold: float = 40.0
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
    head_duration: float = 0.0  # 고개 기반 집중 저하 지속 시간
    body_duration: float = 0.0  # 몸 기울기 기반 집중 저하 지속 시간
    no_face_duration: float = 0.0 # 얼굴 미검출 지속 시간
    face_detected: bool = False
    smoothed_yaw: float = 0.0
    smoothed_pitch: float = 0.0
    smoothed_roll: float = 0.0
    smoothed_body_tilt: float = 0.0
    gaze_direction: str = "Unknown"
    blink_bpm: int = 0
    eye_focus_score: float = 100.0
    eye_status_msg: str = "Eye analysis disabled"

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
    
    def _smooth_body_tilt(self, body_tilt: float) -> None:
        self.state.smoothed_body_tilt = self._smooth_value(
        self.state.smoothed_body_tilt,
        body_tilt,
    )

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

    def _is_screen_fixated(self) -> bool:
        """머리 방향과 눈 방향을 함께 보고 실제로 화면을 응시 중인지 판단한다.

        - 눈이 중앙이면 기본적으로 화면 응시로 본다.
        - 고개가 한쪽으로 돌아갔더라도, 눈이 반대 방향으로 보상하면
          계속 화면(정면)을 응시하는 상황으로 해석한다.
        """
        gaze = self.state.gaze_direction
        yaw = self.state.smoothed_yaw

        if gaze == "Center":
            return True

        head_left = yaw < -self.config.focused_yaw_threshold
        head_right = yaw > self.config.focused_yaw_threshold
        eye_left = gaze == "Right"
        eye_right = gaze == "Left"

        if head_left and eye_right:
            return True
        if head_right and eye_left:
            return True

        return False

    # 집중 저하 지속 시간 업데이트: 얼굴이 감지되고, 일정 각도 이상으로 고개가 돌아간 경우 지속 시간 증가, 그렇지 않고 집중 범위에 있으면 지속 시간 감소
    def _update_distracted_duration(self, dt: float) -> None:
        head_warning_event = self._is_over_threshold() and not self._is_screen_fixated()
        body_warning_event = abs(self.state.smoothed_body_tilt) > 20.0

        # head duration tracking
        if head_warning_event:
            self.state.head_duration += dt
        elif self._is_focused_range():
            self.state.head_duration = max(
                0.0,
                self.state.head_duration - dt * self.config.recovery_speed,
            )

        # body duration tracking
        if body_warning_event:
            self.state.body_duration += dt
        else:
            self.state.body_duration = max(
                0.0,
                self.state.body_duration - dt * self.config.recovery_speed,
            )

    def _calculate_score(self) -> float:
        # 몸 기울기 패널티: 기울기 상태가 얼마나 오래 지속되는지에 따라 증가(최대 3초까지 패널티 적용), 패널티는 최대 (3.0*5.0)15점까지 적용
        body_penalty = min(self.state.body_duration, 3.0) * 5.0

        # 지속 시간 패널티: 고개와 몸의 집중 저하 지속 시간 중 더 긴 쪽을 기준으로 패널티 계산, 최대 5초까지 패널티 적용
        combined_duration = max(self.state.head_duration, self.state.body_duration)
        duration_penalty = min(combined_duration, 5.0) * 8.0
        no_face_penalty = min(self.state.no_face_duration, 3.0) * 12.0

        # gaze 방향 패널티 (핵심)
        # 단순히 눈이 Center가 아닌지만 보지 않고,
        # head pose와 함께 실제 화면 응시(screen fixation) 여부를 해석한다.
        gaze_penalty = 0.0
        if not self._is_screen_fixated():
            gaze_penalty = 25.0  # 실제 화면 응시가 아니면 강하게 감점

        # eye_focus_score 기반 보조 패널티
        eye_penalty = (100.0 - self.state.eye_focus_score) * 0.2

        # 100점에서 각 패널티를 차감하여 최종 점수 계산, 점수는 0에서 100 사이로 제한
        score = (
            100.0
            - body_penalty
            - duration_penalty
            - no_face_penalty
            - gaze_penalty      # 시선 방향 패널티 추가
            - eye_penalty       # eye_focus_score 기반 패널티 추가
        )

        return max(0.0, min(100.0, score))

    def _update_state_label(self) -> None:
        # 1) 얼굴이 일정 시간 이상 검출되지 않으면 ABSENT
        if self.state.no_face_duration >= self.config.no_face_time:
            self.state.state = "ABSENT"
            return

        # 얼굴이 아직 돌아오지 않았지만 ABSENT 임계치 전이라면 우선 LOST_FOCUS로 본다.
        if not self.state.face_detected:
            self.state.state = "LOST_FOCUS"
            return

        combined_duration = max(self.state.head_duration, self.state.body_duration)
        screen_fixated = self._is_screen_fixated()
        body_unstable = abs(self.state.smoothed_body_tilt) > 20.0
        eye_degraded = self.state.eye_focus_score < 70.0

        # 2) 화면 응시가 깨진 상태가 충분히 지속되면 LOST_FOCUS
        if not screen_fixated:
            if combined_duration >= self.config.distracted_time:
                self.state.state = "LOST_FOCUS"
            else:
                # 짧은 비응시 구간은 회복/전환 구간으로 보고 PARTIAL_FOCUS로 둔다.
                self.state.state = "PARTIAL_FOCUS"
            return

        # 3) 화면은 보고 있지만 자세/눈 상태가 불안정하면 PARTIAL_FOCUS
        if body_unstable or eye_degraded:
            self.state.state = "PARTIAL_FOCUS"
            return

        # 4) 나머지는 FOCUSED
        self.state.state = "FOCUSED"

    # 업데이트 메서드: 얼굴 감지 여부와 각도 정보를 받아 상태를 업데이트하고, 최종적으로 현재 상태를 반환
    def update(
        self,
        pose_angles: PoseAngles,
        body_tilt: float,
        face_detected: bool,
        dt: float,
        gaze_direction: str = "Unknown",
        blink_bpm: int = 0,
        eye_focus_score: float = 100.0,
        eye_status_msg: str = "Eye analysis disabled",
    ) -> AttentionState:
        self.state.face_detected = face_detected
        self.state.gaze_direction = gaze_direction
        self.state.blink_bpm = blink_bpm
        self.state.eye_focus_score = eye_focus_score
        self.state.eye_status_msg = eye_status_msg

        if face_detected:
            self.state.no_face_duration = 0.0
            
            self._smooth_pose(pose_angles)
            self._smooth_body_tilt(body_tilt)

            self._update_distracted_duration(dt)
        else:
            self.state.no_face_duration += dt
            self.state.head_duration += dt
            self.state.body_duration += dt

        self.state.score = self._calculate_score()
        self._update_state_label()
        return self.state
