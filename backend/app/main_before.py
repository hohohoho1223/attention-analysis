from __future__ import annotations

import os
# 🌟 [가장 중요] 최신 텐서플로우에게 옛날(Keras 2) 엔진을 사용하라고 강제 명령! (무조건 맨 위에 있어야 합니다)
# ai_serveㅇr/main.py에서 설정한 것과 동일하게 환경변수를 설정하여 TensorFlow가 레거시 Keras 엔진을 사용하도록 강제합니다. 이렇게 하면 모델 호환성 문제를 방지할 수 있습니다.
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from typing import Optional
from collections import deque

import argparse
import mediapipe as mp
import numpy as np
from ai.attention.analyzers.attention_logic import AttentionAnalyzer, AttentionConfig, DEFAULT_ATTENTION_CONFIG
from ai.attention.analyzers.eye_focus import EyeFocusAnalyzer
from ai.attention.analyzers.head_pose import HeadPoseEstimator, PoseAngles
from ai.attention.analyzers.upperbody_pose import UPPER_BODY_LANDMARKS, UpperBodyAnalyzer, UpperBodyState

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
        save_path: Optional[str] = None,
        max_num_faces: int = 1, # 최대 1명 인식
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        process_scale: float = 0.75,
        draw_tesselation: bool = False,
        draw_head_pose_indices: bool = False,
        draw_upper_body_indices: bool = False,
        enable_defense_model: bool = True,
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

        # 🌟 1. CNN 방어막 모델 안전하게 로드
        self.defense_model = None
        self.tf = None
        self.cnn_prob_buffer = deque(maxlen=8)  # 시간 윈도우 스무딩
        
        # 건우님이 구워오신 모델명!
        model_name = 'face_defense_model.h5' 
        model_path = Path(model_name)

        if not self.enable_defense_model:
            print("[INFO] CNN 방어막 모델 비활성화 상태로 실행합니다.")
        elif not model_path.exists():
            print(f"[WARN] '{model_name}' 파일을 찾을 수 없어 CNN 방어막 없이 실행합니다.")
        else:
            print(f"🛡️ AI 방어막 모델({model_name})을 로드하는 중...")
        
            try:
                import tensorflow as tf

                self.tf = tf
                try:
                    self.defense_model = tf.keras.models.load_model(model_path, compile=False)
                except Exception:
                    self.defense_model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        safe_mode=False,
                    )
                print("✅ 1티어 방어막 모델 로드 완료!")
            except Exception as e:
                self.defense_model = None
                self.tf = None
                print(f"❌ 모델 로드 실패. CNN 방어막 없이 계속 실행합니다: {e}")

        # MediaPipe 초기화
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
        debug_points = {"nose_tip": 1, "chin": 152, "left_eye_outer": 263, "right_eye_outer": 33, "left_mouth": 291, "right_mouth": 61}
        for name, idx in debug_points.items():
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{name}:{idx}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # 🌟 2. 얼굴 Bounding Box 추출
    def _get_crop_img(self, frame, face_landmarks, indices, w, h, padding=15):
        """MediaPipe 랜드마크를 기반으로 눈/입술 특정 부위만 쏙 잘라옵니다."""
        x_coords = [int(face_landmarks.landmark[i].x * w) for i in indices]
        y_coords = [int(face_landmarks.landmark[i].y * h) for i in indices]
        x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
        y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)

        if x_max <= x_min or y_max <= y_min: 
            return None

        crop = frame[y_min:y_max, x_min:x_max]
        img_resized = cv2.resize(crop, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 🚨 [버그 수정] 전처리(preprocess_input) 중복 제거! (모델 안에 이미 내장됨)
        return img_rgb

    # 🌟 3. CNN 모델 추론 및 스무딩
    def _check_cnn_state(self, frame, face_landmarks):
        """눈, 입술을 각각 잘라 CNN에 넣고 하나라도 비정상이면 알람을 울립니다."""
        if self.defense_model is None:
            return False
            
        h, w = frame.shape[:2]
        MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

        # 🌟 입술(하관) 크롭 패딩을 20 -> 60으로 대폭 늘려 훈련 데이터와 환경을 맞춥니다!
        left_eye = self._get_crop_img(frame, face_landmarks, self.LEFT_EYE, w, h, padding=15)
        right_eye = self._get_crop_img(frame, face_landmarks, self.RIGHT_EYE, w, h, padding=15)
        mouth = self._get_crop_img(frame, face_landmarks, MOUTH_INDICES, w, h, padding=60)

        imgs = [img for img in [left_eye, right_eye, mouth] if img is not None]
        if not imgs:
            return False

        img_array = np.array(imgs).astype(np.float32)
        predictions = self.defense_model.predict(img_array, verbose=0)
        
        # 🌟 터미널에 실시간으로 확률을 찍어봅니다. (어디가 문제인지 1초 만에 파악 가능!)
        if len(predictions) == 3:
            print(f"[CNN 분석] 좌안: {predictions[0][1]*100:.1f}% | 우안: {predictions[1][1]*100:.1f}% | 입(하관): {predictions[2][1]*100:.1f}%")

        max_abnormal_prob = float(np.max(predictions[:, 1]))
        
        self.cnn_prob_buffer.append(max_abnormal_prob)
        smooth_prob = sum(self.cnn_prob_buffer) / len(self.cnn_prob_buffer)
        
        # 🌟 임계치를 0.85로 높여서 진짜 확실하게 졸거나 하품할 때만 잡게 만듭니다.
        return smooth_prob > 0.77

    def draw_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_for_inference = cv2.resize(rgb, None, fx=self.process_scale, fy=self.process_scale, interpolation=cv2.INTER_LINEAR) if 0 < self.process_scale < 1.0 else rgb

        results = self.face_mesh.process(rgb_for_inference)
        self.face_detected = bool(results.multi_face_landmarks)

        pose_results = self.pose.process(rgb)
        if pose_results.pose_landmarks:
            self.upper_body_state = self.upper_body_analyzer.estimate(pose_results.pose_landmarks, frame.shape[1], frame.shape[0])
        else:
            self.upper_body_state = UpperBodyState()

        if self.draw_upper_body_indices and self.upper_body_state.body_visible:
            display_names = {"left_shoulder": "L_shoulder", "right_shoulder": "R_shoulder", "left_elbow": "L_elbow", "right_elbow": "R_elbow"}
            for name, (x, y) in self.upper_body_state.landmark_points.items():
                idx = UPPER_BODY_LANDMARKS[name]
                label = display_names[name]
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{label}:{idx}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        if results.multi_face_landmarks:
            self.current_face_landmarks = results.multi_face_landmarks[0]
            for face_landmarks in results.multi_face_landmarks:
                self._update_head_pose_state(face_landmarks, frame)
                eye_result = self.eye_focus_analyzer.analyze(frame, face_landmarks)
                self.gaze_direction = eye_result.gaze_direction
                self.blink_bpm = eye_result.blink_bpm
                self.eye_focus_score = eye_result.eye_focus_score
                self.eye_status_msg = eye_result.eye_status_msg

                if self.draw_head_pose_indices: self._draw_head_pose_landmark_indices(frame, face_landmarks)
                if self.draw_tesselation:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                if self.refine_landmarks:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            cv2.putText(frame, f"MediaPipe Face Mesh | faces: {len(results.multi_face_landmarks)} | scale: {self.process_scale:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {self.pose_angles.yaw:.1f} | Pitch: {self.pose_angles.pitch:.1f} | Roll: {self.pose_angles.roll:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        else:
            self.current_face_landmarks = None
            self.pose_angles = PoseAngles()
            self.face_detected = False
            self.eye_focus_analyzer.reset()
            self.gaze_direction = "Unknown"
            self.blink_bpm = 0
            self.eye_focus_score = 0.0
            self.eye_status_msg = "Face Not Detected (Eye disabled)"
            cv2.putText(frame, f"MediaPipe Face Mesh | no face | scale: {self.process_scale:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        if self.upper_body_state.body_visible:
            cv2.putText(frame, f"Shoulder Tilt: {self.upper_body_state.shoulder_tilt:.1f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 0), 2)

        return frame

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
        output = self.draw_mediapipe(frame)

        # 🌟 4. CNN 방어막 가동
        is_drowsy = False
        if self.face_detected and self.current_face_landmarks:
            is_drowsy = self._check_cnn_state(frame, self.current_face_landmarks)

        try:
            attention_state = self.attention_analyzer.update(
                pose_angles=self.pose_angles,
                body_tilt=self.upper_body_state.shoulder_tilt,
                face_detected=self.face_detected,
                dt=1.0 / max(self.fps, 1e-6),
                gaze_direction=self.gaze_direction,
                blink_bpm=self.blink_bpm,
                eye_focus_score=self.eye_focus_score,
                eye_status_msg=self.eye_status_msg,
                is_drowsy=is_drowsy
            )
        except TypeError:
            attention_state = self.attention_analyzer.update(
                pose_angles=self.pose_angles,
                body_tilt=self.upper_body_state.shoulder_tilt,
                face_detected=self.face_detected,
                dt=1.0 / max(self.fps, 1e-6),
                gaze_direction=self.gaze_direction,
                blink_bpm=self.blink_bpm,
                eye_focus_score=self.eye_focus_score,
                eye_status_msg=self.eye_status_msg
            )
            if is_drowsy:
                cv2.putText(output, "🚨 DROWSY (Update logic!)", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # UI 출력부
        state_color = (0, 0, 255) if getattr(attention_state, 'state', '') == "DROWSY" else (255, 180, 0)
        cv2.putText(output, f"Attention: {attention_state.state} | Score: {attention_state.score:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(output, f"Head: {attention_state.head_duration:.1f}s | Body: {attention_state.body_duration:.1f}s", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
        cv2.putText(output, f"Gaze: {self.gaze_direction} | BPM: {self.blink_bpm} | EyeScore: {self.eye_focus_score:.1f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

        fixation_status = "FIXATED" if self.attention_analyzer._is_screen_fixated() else "NOT FIXATED"

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
        # ===== DEBUG LOG (원인 + 결과) =====
        screen_fixated = self.attention_analyzer._is_screen_fixated()

        yaw = attention_state.smoothed_yaw
        if yaw > self.attention_config.focused_yaw_threshold:
            head_direction = "Left"
        elif yaw < -self.attention_config.focused_yaw_threshold:
            head_direction = "Right"
        else:
            head_direction = "Center"

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

        # 상태 변화 로그
        if not hasattr(self, "prev_state"):
            self.prev_state = None

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
        if self.writer is not None: self.writer.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

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

# 이 엔진이 없어서 아까 소리 소문 없이 끝난 겁니다! 😂
if __name__ == "__main__":
    main()