from __future__ import annotations

from dataclasses import dataclass

from ai.attention.analyzers.head_pose import PoseAngles


@dataclass(frozen=True)
class AttentionConfig:
    yaw_threshold: float = 60.0 # 확실히 돌아간 상태로 간주하는 각도 임계치
    pitch_threshold: float = 30.0
    roll_threshold: float = 25.0
    focused_yaw_threshold: float = 33.0 # 정면 허용 범위 각도 -> 이 범위 안이면 '정면(집중 가능한 상태)'으로 본다
    focused_pitch_threshold: float = 15.0
    focused_roll_threshold: float = 10.0
    partial_focus_time: float = 3.0
    lost_focus_time: float = 5.0
    no_face_time: float = 10.0
    smoothing_alpha: float = 0.35 # 스무딩 가중치 0.35(0.0 ~ 1.0)로 설정-> 현재 측정값이 35%, 이전 측정값 65% 반영
    recovery_speed: float = 1.5


DEFAULT_ATTENTION_CONFIG = AttentionConfig()


@dataclass
class AttentionState:
    state: str = "FOCUSED"
    score: float = 100.0
    head_duration: float = 0.0  # 고개 기반 집중 저하 지속 시간
    body_duration: float = 0.0  # 몸 기울기 기반 집중 저하 지속 시간
    fixation_break_duration: float = 0.0  # 실제 화면 비응시 지속 시간 (head + eye 종합)
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
    absent_count: int = 0

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
        """머리 방향 + 눈 방향을 함께 고려한 실제 화면 응시 판단

        핵심:
        - gaze == Center만으로는 응시로 보지 않는다.
        - head가 돌아갔으면 반드시 eye가 반대 방향으로 보상해야 한다.
        """
        gaze = self.state.gaze_direction
        yaw = self.state.smoothed_yaw

        # 현재 프로젝트의 yaw 부호 기준에 맞춘 해석
        # - 사용자가 화면 기준 오른쪽으로 고개를 돌리면 yaw가 음수
        # - 사용자가 화면 기준 왼쪽으로 고개를 돌리면 yaw가 양수
        head_left = yaw > self.config.focused_yaw_threshold
        head_right = yaw < -self.config.focused_yaw_threshold

        eye_left = gaze == "Left"
        eye_right = gaze == "Right"
        eye_center = gaze == "Center"

        # 1. head가 정면일 때만 center 허용
        if not head_left and not head_right:
            return eye_center

        # 2. head가 돌아갔으면 eye는 반대 방향이어야 함 (보상)
        if head_left and eye_right:
            return True
        if head_right and eye_left:
            return True

        # 3. 그 외는 모두 비응시
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

        # fixation break duration tracking
        screen_fixated = self._is_screen_fixated()
        if not screen_fixated:
            self.state.fixation_break_duration += dt
        else:
            self.state.fixation_break_duration = max(
                0.0,
                self.state.fixation_break_duration - dt * self.config.recovery_speed,
            )

    def _calculate_score(self) -> float:
        """상태(state)를 먼저 해석하고, 그 상태가 얼마나 지속되었는지로 점수를 보정한다.

        철학:
        - 상태가 먼저다: FOCUSED / PARTIAL_FOCUS / LOST_FOCUS / ABSENT
        - 지속시간은 같은 상태 안에서 강도를 결정한다.
        - feature별 즉시 감점 합산보다 상태 기반 해석을 우선한다.
        """
        combined_duration = max(self.state.head_duration, self.state.body_duration)
        partial_driver = max(self.state.body_duration, self.state.fixation_break_duration)
        lost_driver = max(self.state.fixation_break_duration, self.state.no_face_duration)

        if self.state.state == "ABSENT":
            # 이탈 상태는 가장 낮은 점수 구간에서 빠르게 0점으로 수렴
            score = 20.0 - min(self.state.no_face_duration, 3.0) * 7.0
        elif self.state.state == "LOST_FOCUS":
            # 화면 응시가 깨진 상태. 지속될수록 55 → 30 정도까지 하락
            score = 55.0 - min(lost_driver, 5.0) * 5.0
        elif self.state.state == "DROWSY":
            # 점수 급락 방지 (완화된 감소)
            score = max(30.0, 60.0 - self.drowsy_duration * 10.0)
        elif self.state.state == "PARTIAL_FOCUS":
            # 화면은 보고 있지만 자세/눈 상태가 불안정한 상태
            score = 80.0 - min(partial_driver, 4.0) * 3.0

            # eye 상태가 조금 나쁘면 같은 PARTIAL_FOCUS 안에서 약하게 추가 보정
            if self.state.face_detected and self.state.eye_status_msg != "Eye analysis disabled":
                safe_eye_score = max(0.0, min(100.0, self.state.eye_focus_score))
                score -= (100.0 - safe_eye_score) * 0.05  # 완화된 패널티
        else:
            # FOCUSED는 점수 고정 (누적 감점 방지)
            score = 100.0

        return max(0.0, min(100.0, score))

    def _update_state_label(self) -> None:
        # 1) 얼굴이 일정 시간 이상 검출되지 않으면 ABSENT
        if self.state.no_face_duration >= self.config.no_face_time:
            # ABSENT 상태 진입 시 카운트 증가 (중복 증가 방지)
            if self.state.state != "ABSENT":
                self.state.absent_count += 1
            self.state.state = "ABSENT"
            return

        # 얼굴 미검출 → duration 기반 단계 처리
        if not self.state.face_detected:
            if self.state.no_face_duration >= self.config.no_face_time:
                if self.state.state != "ABSENT":
                    self.state.absent_count += 1
                self.state.state = "ABSENT"
            elif self.state.no_face_duration >= 3.0:
                self.state.state = "LOST_FOCUS"
            else:
                self.state.state = "PARTIAL_FOCUS"
            return
        
        # --- DROWSY 판단 (모델 결과 기반 단순화) ---
        # 졸음 상태 감지 시 drowsy_duration이 증가하고, 일정 시간 이상 지속되면 DROWSY 상태로 진입
        if self.drowsy_duration > 3.0:
            self.state.state = "DROWSY"
            return
        
        screen_fixated = self._is_screen_fixated()
        body_unstable = abs(self.state.smoothed_body_tilt) > 20.0
        eye_degraded = self.state.eye_focus_score < 70.0
        fixation_break = self.state.fixation_break_duration

        # 2) 화면 비응시 상태(NOT_fixated)는 지속시간으로 판단한다.
        #    - 아주 짧으면 FOCUSED 유지 (노이즈 / 회복 구간 고려)
        #    - 조금 지속되면 PARTIAL_FOCUS
        #    - 오래 지속되면 LOST_FOCUS
        if not screen_fixated:
            if fixation_break >= self.config.lost_focus_time:
                self.state.state = "LOST_FOCUS"
            elif fixation_break >= self.config.partial_focus_time:
                self.state.state = "PARTIAL_FOCUS"
            else:
                self.state.state = "FOCUSED"
            return

        # 3) 화면 응시 상태(fixated)라도, 몸이 많이 기울어졌거나 눈 상태가 많이 나쁘면 PARTIAL_FOCUS
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
        is_drowsy: bool = False
    ) -> AttentionState:
        self.state.face_detected = face_detected
        self.state.gaze_direction = gaze_direction
        self.state.blink_bpm = blink_bpm
        self.state.eye_focus_score = eye_focus_score
        self.state.eye_status_msg = eye_status_msg

        # calibration 상태 감지 (얼굴이 보이는 상태에서만 수행)
        if face_detected and gaze_direction == "Unknown" and self.state.no_face_duration < 1.0:
            self.state.state = "CALIBRATING"
            self.state.is_fixated = False
            return self.state

        # --- DROWSY evidence 누적 ---
        if not hasattr(self, "drowsy_duration"):
            self.drowsy_duration = 0.0

        if is_drowsy:
            self.drowsy_duration += dt * 1.0  # drowsy 상태가 감지되면 지속 시간 증가
        else:
            self.drowsy_duration = max(0.0, self.drowsy_duration - dt * 1.5)

        if face_detected:
            self.state.no_face_duration = 0.0
            
            self._smooth_pose(pose_angles)
            self._smooth_body_tilt(body_tilt)

            self._update_distracted_duration(dt)
        else:
            self.state.no_face_duration += dt
            self.state.head_duration += dt
            self.state.body_duration += dt

        self._update_state_label()
        self.state.score = self._calculate_score()
        self.state.is_fixated = self._is_screen_fixated()
        return self.state
