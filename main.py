from __future__ import annotations

import argparse
from typing import Optional

import mediapipe as mp
from attention_logic import AttentionAnalyzer, AttentionConfig, DEFAULT_ATTENTION_CONFIG
from eye_focus import EyeFocusAnalyzer
from head_pose import HeadPoseEstimator, PoseAngles
from upperbody_pose import UPPER_BODY_LANDMARKS, UpperBodyAnalyzer, UpperBodyState


try:
    import cv2
except Exception as exc:
    raise RuntimeError(
        "OpenCV(cv2) import 에 실패했습니다. 현재 환경의 opencv-python 버전 충돌 가능성이 큽니다.\n"
        "requirements.txt 기준으로 새 가상환경에서 다시 설치하는 것을 권장합니다.\n"
        "예시:\n"
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt"
    ) from exc

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

DEFAULT_RUNTIME_CONFIG = DEFAULT_ATTENTION_CONFIG


class VideoFaceAnalyzer:
    def __init__(
        self,
        source: str,
        save_path: Optional[str] = None,
        max_num_faces: int = 1, # 최대 1명 인식
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        process_scale: float = 0.75,
        draw_tesselation: bool = False,
        draw_head_pose_indices: bool = False,
        draw_upper_body_indices: bool = False,
        attention_config: AttentionConfig = DEFAULT_RUNTIME_CONFIG,
    ) -> None:
        self.source = 0 if source == "webcam" else source
        self.save_path = save_path
        self.max_num_faces = max_num_faces
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.refine_landmarks = refine_landmarks
        self.process_scale = process_scale
        self.draw_tesselation = draw_tesselation
        self.draw_head_pose_indices = draw_head_pose_indices
        self.draw_upper_body_indices = draw_upper_body_indices
        self.attention_config = attention_config

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {source}")

        self.writer = None
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.is_video_file = self.source != 0
        self.is_paused = False
        self.frame_delay_ms = max(1, int(round(1000 / self.fps)))
        self.current_frame_idx = -1
        self.pose_angles = PoseAngles()
        self.face_detected = False
        self.gaze_direction = "Unknown"
        self.blink_bpm = 0
        self.eye_focus_score = 100.0
        self.eye_status_msg = "Eye analysis disabled"

        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.save_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.head_pose_estimator = HeadPoseEstimator()
        self.upper_body_analyzer = UpperBodyAnalyzer()
        self.upper_body_state = UpperBodyState()

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        self.attention_analyzer = AttentionAnalyzer(config=self.attention_config)
        self.eye_focus_analyzer = EyeFocusAnalyzer()

    def _update_head_pose_state(self, face_landmarks, frame) -> None:
        frame_height, frame_width = frame.shape[:2]
        pose = self.head_pose_estimator.estimate(face_landmarks, frame_width, frame_height)
        if pose is not None:
            self.pose_angles = pose

    def _draw_head_pose_landmark_indices(self, frame, face_landmarks) -> None:
        frame_height, frame_width = frame.shape[:2]
        debug_points = {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_outer": 263,
            "right_eye_outer": 33,
            "left_mouth": 291,
            "right_mouth": 61,
        }

        for name, idx in debug_points.items():
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{name}:{idx}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

    def draw_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if 0 < self.process_scale < 1.0:
            rgb_for_inference = cv2.resize(
                rgb,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            rgb_for_inference = rgb

        results = self.face_mesh.process(rgb_for_inference)
        self.face_detected = bool(results.multi_face_landmarks)

        # Pose estimation for upper body
        pose_results = self.pose.process(rgb)
        if pose_results.pose_landmarks:
            self.upper_body_state = self.upper_body_analyzer.estimate(
                pose_results.pose_landmarks,
                frame.shape[1],
                frame.shape[0],
            )
        else:
            self.upper_body_state = UpperBodyState()

        if self.draw_upper_body_indices and self.upper_body_state.body_visible:
            display_names = {
                "left_shoulder": "L_shoulder",
                "right_shoulder": "R_shoulder",
                "left_elbow": "L_elbow",
                "right_elbow": "R_elbow",
            }

            for name, (x, y) in self.upper_body_state.landmark_points.items():
                idx = UPPER_BODY_LANDMARKS[name]
                label = display_names[name]

                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"{label}:{idx}",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._update_head_pose_state(face_landmarks, frame)

                # 얼굴 있을 때 eye 분석 연동
                eye_result = self.eye_focus_analyzer.analyze(frame, face_landmarks)
                self.gaze_direction = eye_result.gaze_direction
                self.blink_bpm = eye_result.blink_bpm
                self.eye_focus_score = eye_result.eye_focus_score
                self.eye_status_msg = eye_result.eye_status_msg

                if self.draw_head_pose_indices:
                    self._draw_head_pose_landmark_indices(frame, face_landmarks)

                if self.draw_tesselation:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                if self.refine_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

            cv2.putText(
                frame,
                f"MediaPipe Face Mesh | faces: {len(results.multi_face_landmarks)} | scale: {self.process_scale:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                (
                    f"Yaw: {self.pose_angles.yaw:.1f} | "
                    f"Pitch: {self.pose_angles.pitch:.1f} | "
                    f"Roll: {self.pose_angles.roll:.1f}"
                ),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
        else:
            self.pose_angles = PoseAngles()
            self.face_detected = False

            # 얼굴 없을 때 eye 완전 비활성화 (face dependency 강제)
            self.eye_focus_analyzer.reset()
            self.gaze_direction = "Unknown"
            self.blink_bpm = 0
            self.eye_focus_score = 0.0
            self.eye_status_msg = "Face Not Detected (Eye disabled)"

            cv2.putText(
                frame,
                f"MediaPipe Face Mesh | no face | scale: {self.process_scale:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            
        # 상반신 상태 표시
        if self.upper_body_state.body_visible:
            cv2.putText(
                frame,
                f"Shoulder Tilt: {self.upper_body_state.shoulder_tilt:.1f}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 255, 0),
                2,
            )

        return frame

    def seek_frames(self, offset_frames: int) -> None:
        if not self.is_video_file:
            return

        base_frame = self.current_frame_idx
        if base_frame < 0:
            base_frame = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        target_frame = base_frame + offset_frames

        if self.total_frames > 0:
            target_frame = max(0, min(target_frame, self.total_frames - 1))
        else:
            target_frame = max(0, target_frame)

        self.current_frame_idx = target_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def show_current_frame(self) -> None:
        if not self.is_video_file or self.current_frame_idx < 0:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        output = self.process_frame(frame)
        cv2.imshow("Face Analyzer", output)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx + 1)

    def process_frame(self, frame):
        output = self.draw_mediapipe(frame)

        attention_state = self.attention_analyzer.update(
        pose_angles=self.pose_angles,
        body_tilt=self.upper_body_state.shoulder_tilt,
        face_detected=self.face_detected,
        dt=1.0 / max(self.fps, 1e-6),
        gaze_direction=self.gaze_direction,
        blink_bpm=self.blink_bpm,
        eye_focus_score=self.eye_focus_score,
        eye_status_msg=self.eye_status_msg,
        )

        cv2.putText(
            output,
            (
                f"Attention: {attention_state.state} | "
                f"Score: {attention_state.score:.1f}"
            ),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2,
        )
        cv2.putText(
            output,
            (
                f"Head: {attention_state.head_duration:.1f}s | "
                f"Body: {attention_state.body_duration:.1f}s"
            ),
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2,
        )
        cv2.putText(
            output,
            (
                f"Gaze: {attention_state.gaze_direction} | "
                f"BPM: {attention_state.blink_bpm} | "
                f"EyeScore: {attention_state.eye_focus_score:.1f}"
            ),
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )
        cv2.putText(
            output,
            f"EyeStatus: {attention_state.eye_status_msg}",
            (10, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            2,
        )

        status_text = "PAUSED" if self.is_paused else "PLAYING"
        cv2.putText(
            output,
            f"{status_text} | fps: {self.fps:.1f} | q: quit | space: pause | a/d: -/+5s | j/l: -/+1s",
            (10, self.frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return output

    def handle_key(self, key: int) -> bool:
        if key == ord("q"):
            print("[INFO] 사용자 종료")
            return False

        if key == ord(" "):
            self.is_paused = not self.is_paused
            print(f"[INFO] {'일시정지' if self.is_paused else '재생'}")
            return True

        if key == ord("a"):
            self.seek_frames(-int(self.fps * 5))
            if self.is_paused:
                self.show_current_frame()
            return True

        if key == ord("d"):
            self.seek_frames(int(self.fps * 5))
            if self.is_paused:
                self.show_current_frame()
            return True

        if key == ord("j"):
            self.seek_frames(-int(self.fps))
            if self.is_paused:
                self.show_current_frame()
            return True

        if key == ord("l"):
            self.seek_frames(int(self.fps))
            if self.is_paused:
                self.show_current_frame()
            return True

        return True

    def run(self) -> None:
        print("[INFO] 실행 시작")
        print("[INFO] 단축키: q 종료 | space 일시정지/재생 | a/d 5초 뒤로/앞으로 | j/l 1초 뒤로/앞으로")

        while True:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] 영상 끝 또는 프레임을 읽지 못했습니다. 종료합니다.")
                    break

                self.current_frame_idx = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                output = self.process_frame(frame)
                cv2.imshow("Face Analyzer", output)

                if self.writer is not None:
                    self.writer.write(output)

            wait_time = 30 if self.is_paused else self.frame_delay_ms
            key = cv2.waitKey(wait_time) & 0xFF
            if key != 255:
                should_continue = self.handle_key(key)
                if not should_continue:
                    break

        self.release()

    def release(self) -> None:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
# main.py는 실행/입출력만 담당합니다.
# 집중도 판별 기준값의 기본 원본은 attention_logic.py의 DEFAULT_ATTENTION_CONFIG 입니다.
# threshold/time 값을 바꾸려면 AttentionConfig 또는 DEFAULT_ATTENTION_CONFIG 쪽을 수정하면 됩니다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Mesh 기반 고개 움직임(Yaw/Pitch/Roll) 분석 예제"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="입력 소스. webcam 또는 영상 파일 경로",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="결과 영상 저장 경로 (예: output.mp4)",
    )
    parser.add_argument(
        "--max-num-faces",
        type=int,
        default=1,
        help="MediaPipe에서 추적할 최대 얼굴 수",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe 최소 탐지 신뢰도",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe 최소 추적 신뢰도",
    )
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        help="정밀 랜드마크(홍채 포함) 사용. 정확도는 조금 좋아질 수 있지만 더 느립니다.",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=0.75,
        help="MediaPipe 추론 전 프레임 축소 비율. 0.5~1.0 권장",
    )
    parser.add_argument(
        "--draw-tesselation",
        action="store_true",
        help="Face Mesh 촘촘한 삼각형 선까지 그림. 더 느려질 수 있습니다.",
    )
    parser.add_argument(
        "--draw-head-pose-indices",
        action="store_true",
        help="head pose에 사용하는 랜드마크 인덱스 번호를 화면에 표시합니다.",
    )
    parser.add_argument(
        "--draw-upper-body-indices",
        action="store_true",
        help="어깨(11,12)와 팔꿈치(13,14) MediaPipe Pose 랜드마크 인덱스를 화면에 표시합니다.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    attention_config = AttentionConfig(
        yaw_threshold=DEFAULT_RUNTIME_CONFIG.yaw_threshold,
        pitch_threshold=DEFAULT_RUNTIME_CONFIG.pitch_threshold,
        roll_threshold=DEFAULT_RUNTIME_CONFIG.roll_threshold,
        focused_yaw_threshold=DEFAULT_RUNTIME_CONFIG.focused_yaw_threshold,
        focused_pitch_threshold=DEFAULT_RUNTIME_CONFIG.focused_pitch_threshold,
        focused_roll_threshold=DEFAULT_RUNTIME_CONFIG.focused_roll_threshold,
        minor_distraction_time=DEFAULT_RUNTIME_CONFIG.minor_distraction_time,
        distracted_time=DEFAULT_RUNTIME_CONFIG.distracted_time,
        lost_focus_time=DEFAULT_RUNTIME_CONFIG.lost_focus_time,
        no_face_time=DEFAULT_RUNTIME_CONFIG.no_face_time,
        smoothing_alpha=DEFAULT_RUNTIME_CONFIG.smoothing_alpha,
        recovery_speed=DEFAULT_RUNTIME_CONFIG.recovery_speed,
    )

    analyzer = VideoFaceAnalyzer(
        source=args.source,
        save_path=args.save_path,
        max_num_faces=args.max_num_faces,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        refine_landmarks=True,  # 홍채 포함 정밀 랜드마크 항상 활성화
        process_scale=args.process_scale,
        draw_tesselation=args.draw_tesselation,
        draw_head_pose_indices=args.draw_head_pose_indices,
        draw_upper_body_indices=args.draw_upper_body_indices,
        attention_config=attention_config,
    )
    analyzer.run()


if __name__ == "__main__":
    main()