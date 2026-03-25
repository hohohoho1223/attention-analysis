from __future__ import annotations

import argparse
import mediapipe as mp
from attention.analyzers.attention_logic import AttentionConfig, DEFAULT_ATTENTION_CONFIG
from attention.analyzers.head_pose import PoseAngles
from attention.analyzers.upperbody_pose import UPPER_BODY_LANDMARKS, UpperBodyState
from attention.pipeline import AttentionPipeline

try:
    import cv2
except Exception as exc:
    raise RuntimeError(
        "OpenCV(cv2) import 에 실패했습니다. 현재 환경의 opencv-python 버전 충돌 가능성이 큽니다.\n"
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
        save_path: "Optional[str]" = None,
        max_num_faces: int = 1,  # 최대 1명 인식
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        process_scale: float = 0.75,
        draw_tesselation: bool = False,
        draw_head_pose_indices: bool = False,
        draw_upper_body_indices: bool = False,
        enable_defense_model: bool = True,
        attention_config: "AttentionConfig" = DEFAULT_RUNTIME_CONFIG,
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
        self.enable_defense_model = enable_defense_model
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
        self.current_face_landmarks = None

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.save_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

        self.pipeline = AttentionPipeline(
            detection_confidence=self.detection_confidence,
            tracking_confidence=self.tracking_confidence,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            process_scale=self.process_scale,
            enable_defense_model=self.enable_defense_model,
            attention_config=self.attention_config,
        )

        self.upper_body_state = UpperBodyState()
        self.prev_state = None

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
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
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


    def draw_mediapipe(self, frame, result):
        output = frame.copy()
        self.face_detected = result.face_detected
        self.pose_angles = result.pose_angles
        self.gaze_direction = result.gaze_direction
        self.blink_bpm = result.blink_bpm
        self.eye_focus_score = result.eye_focus_score
        self.eye_status_msg = result.eye_status_msg
        self.current_face_landmarks = result.current_face_landmarks
        self.upper_body_state = result.upper_body_state

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
                cv2.circle(output, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(output, f"{label}:{idx}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        if self.current_face_landmarks is not None:
            if self.draw_head_pose_indices:
                self._draw_head_pose_landmark_indices(output, self.current_face_landmarks)
            if self.draw_tesselation:
                mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=self.current_face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            mp_drawing.draw_landmarks(
                image=output,
                landmark_list=self.current_face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            if self.refine_landmarks:
                mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=self.current_face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

            cv2.putText(
                output,
                f"MediaPipe Face Mesh | faces: 1 | scale: {self.process_scale:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                output,
                f"Yaw: {self.pose_angles.yaw:.1f} | Pitch: {self.pose_angles.pitch:.1f} | Roll: {self.pose_angles.roll:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
        else:
            cv2.putText(
                output,
                f"MediaPipe Face Mesh | no face | scale: {self.process_scale:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if self.upper_body_state.body_visible:
            cv2.putText(
                output,
                f"Shoulder Tilt: {self.upper_body_state.shoulder_tilt:.1f}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 255, 0),
                2,
            )

        return output

    def seek_frames(self, offset_frames: int) -> None:
        if not self.is_video_file: return
        base_frame = self.current_frame_idx
        if base_frame < 0: base_frame = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        target_frame = base_frame + offset_frames
        if self.total_frames > 0: target_frame = max(0, min(target_frame, self.total_frames - 1))
        else: target_frame = max(0, target_frame)
        self.current_frame_idx = target_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def show_current_frame(self) -> None:
        if not self.is_video_file or self.current_frame_idx < 0: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret: return
        output = self.process_frame(frame)
        cv2.imshow("Face Analyzer", output)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx + 1)

    def process_frame(self, frame):
        result = self.pipeline.analyze_frame(frame, self.fps)
        output = self.draw_mediapipe(frame, result)

        attention_state = result.attention_state
        state_color = (0, 0, 255) if getattr(attention_state, "state", "") == "DROWSY" else (255, 180, 0)
        cv2.putText(output, f"Attention: {attention_state.state} | Score: {attention_state.score:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(output, f"Head: {attention_state.head_duration:.1f}s | Body: {attention_state.body_duration:.1f}s", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
        cv2.putText(output, f"Gaze: {result.gaze_direction} | BPM: {result.blink_bpm} | EyeScore: {result.eye_focus_score:.1f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

        screen_fixated = self.pipeline.attention_analyzer._is_screen_fixated()
        fixation_status = "FIXATED" if screen_fixated else "NOT FIXATED"

        yaw = attention_state.smoothed_yaw
        if yaw > self.attention_config.focused_yaw_threshold:
            head_direction = "Left"
        elif yaw < -self.attention_config.focused_yaw_threshold:
            head_direction = "Right"
        else:
            head_direction = "Center"

        cv2.putText(
            output,
            f"Fixation: {fixation_status} | HeadDir: {head_direction} | Gaze: {attention_state.gaze_direction}",
            (10, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
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

        print(
            f"[DEBUG] "
            f"State={attention_state.state} | "
            f"Score={attention_state.score:.1f} | "
            f"Fixated={screen_fixated} | "
            f"HeadDir={head_direction}({yaw:.1f}) | "
            f"Gaze={attention_state.gaze_direction} | "
            f"FixBreak={attention_state.fixation_break_duration:.2f} | "
            f"HeadDur={attention_state.head_duration:.2f} | "
            f"BodyDur={attention_state.body_duration:.2f}"
        )

        if self.prev_state != attention_state.state:
            print(
                f"[STATE CHANGE] {self.prev_state} → {attention_state.state} | "
                f"FixBreak={attention_state.fixation_break_duration:.2f} | "
                f"Fixated={screen_fixated}"
            )
            self.prev_state = attention_state.state

        status_text = "PAUSED" if self.is_paused else "PLAYING"
        cv2.putText(output, f"{status_text} | fps: {self.fps:.1f} | q: quit", (10, self.frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return output

    def handle_key(self, key: int) -> bool:
        if key == ord("q"): return False
        if key == ord(" "):
            self.is_paused = not self.is_paused
            return True
        if key == ord("a"):
            self.seek_frames(-int(self.fps * 5))
            if self.is_paused: self.show_current_frame()
            return True
        if key == ord("d"):
            self.seek_frames(int(self.fps * 5))
            if self.is_paused: self.show_current_frame()
            return True
        if key == ord("j"):
            self.seek_frames(-int(self.fps))
            if self.is_paused: self.show_current_frame()
            return True
        if key == ord("l"):
            self.seek_frames(int(self.fps))
            if self.is_paused: self.show_current_frame()
            return True
        return True

    def run(self) -> None:
        print("[INFO] 실행 시작")
        print("[INFO] 단축키: q 종료 | space 일시정지/재생 | a/d 5초 이동 | j/l 1초 이동")
        while True:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret: break
                self.current_frame_idx = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                output = self.process_frame(frame)
                cv2.imshow("Face Analyzer", output)
                if self.writer is not None: self.writer.write(output)

            wait_time = 30 if self.is_paused else self.frame_delay_ms
            key = cv2.waitKey(wait_time) & 0xFF
            if key != 255:
                if not self.handle_key(key): break
        self.release()

    def release(self) -> None:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        self.pipeline.close()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="학습 집중도 분석 시스템")
    parser.add_argument("--source", type=str, default="webcam")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--max-num-faces", type=int, default=1)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--refine-landmarks", action="store_true", default=True)
    parser.add_argument("--process-scale", type=float, default=0.75)
    parser.add_argument("--draw-tesselation", action="store_true")
    parser.add_argument("--draw-head-pose-indices", action="store_true")
    parser.add_argument("--draw-upper-body-indices", action="store_true")
    parser.add_argument("--disable-defense-model", action="store_true")
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
        refine_landmarks=args.refine_landmarks,
        process_scale=args.process_scale,
        draw_tesselation=args.draw_tesselation,
        draw_head_pose_indices=args.draw_head_pose_indices,
        draw_upper_body_indices=args.draw_upper_body_indices,
        enable_defense_model=not args.disable_defense_model,
        attention_config=attention_config,
    )
    analyzer.run()

if __name__ == "__main__":
    main()