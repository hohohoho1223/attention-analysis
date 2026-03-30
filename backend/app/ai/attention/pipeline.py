from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np

from ai.attention.analyzers.attention_logic import AttentionAnalyzer, AttentionConfig
from ai.attention.analyzers.eye_focus import EyeFocusAnalyzer
from ai.attention.analyzers.head_pose import HeadPoseEstimator, PoseAngles
from ai.attention.analyzers.upperbody_pose import UpperBodyAnalyzer, UpperBodyState
from ai.attention.schemas import AttentionFrameResult

try:
    import cv2
except Exception as exc:
    raise RuntimeError(
        "OpenCV(cv2) import 에 실패했습니다. 현재 환경의 opencv-python 버전 충돌 가능성이 큽니다.\n"
    ) from exc

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


class AttentionPipeline:
    """프레임 단위 집중도 분석 파이프라인."""

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        process_scale: float = 0.75,
        enable_defense_model: bool = True,
        attention_config: Optional[AttentionConfig] = None,
    ) -> None:
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.process_scale = process_scale
        self.enable_defense_model = enable_defense_model

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        if attention_config is None:
            raise ValueError("attention_config는 반드시 전달되어야 합니다.")

        self.head_pose_estimator = HeadPoseEstimator()
        self.upper_body_analyzer = UpperBodyAnalyzer()
        self.attention_analyzer = AttentionAnalyzer(config=attention_config)
        self.eye_focus_analyzer = EyeFocusAnalyzer()

        self.defense_model = None
        self.tf = None
        self.cnn_prob_buffer = deque(maxlen=8)
        self._load_defense_model()

    def _load_defense_model(self) -> None:
        model_name = "face_defense_model.h5"
        model_path = Path(model_name)

        if not self.enable_defense_model:
            print("[INFO] CNN 방어막 모델 비활성화 상태로 실행합니다.")
            return

        if not model_path.exists():
            print(f"[WARN] '{model_name}' 파일을 찾을 수 없어 CNN 방어막 없이 실행합니다.")
            return

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

    def _update_head_pose_state(self, face_landmarks, frame) -> PoseAngles:
        frame_height, frame_width = frame.shape[:2]
        pose = self.head_pose_estimator.estimate(face_landmarks, frame_width, frame_height)
        return pose if pose is not None else PoseAngles()

    def _get_crop_img(self, frame, face_landmarks, indices, w, h, padding=15):
        x_coords = [int(face_landmarks.landmark[i].x * w) for i in indices]
        y_coords = [int(face_landmarks.landmark[i].y * h) for i in indices]
        x_min, x_max = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
        y_min, y_max = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)

        if x_max <= x_min or y_max <= y_min:
            return None

        crop = frame[y_min:y_max, x_min:x_max]
        img_resized = cv2.resize(crop, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _check_cnn_state(self, frame, face_landmarks) -> bool:
        if self.defense_model is None:
            return False

        h, w = frame.shape[:2]
        left_eye = self._get_crop_img(frame, face_landmarks, self.LEFT_EYE, w, h, padding=15)
        right_eye = self._get_crop_img(frame, face_landmarks, self.RIGHT_EYE, w, h, padding=15)
        mouth = self._get_crop_img(frame, face_landmarks, self.MOUTH_INDICES, w, h, padding=60)

        imgs = [img for img in [left_eye, right_eye, mouth] if img is not None]
        if not imgs:
            return False

        img_array = np.array(imgs).astype(np.float32)
        predictions = self.defense_model.predict(img_array, verbose=0)

        if len(predictions) == 3:
            print(
                f"[CNN 분석] 좌안: {predictions[0][1] * 100:.1f}% | "
                f"우안: {predictions[1][1] * 100:.1f}% | "
                f"입(하관): {predictions[2][1] * 100:.1f}%"
            )

        max_abnormal_prob = float(np.max(predictions[:, 1]))
        self.cnn_prob_buffer.append(max_abnormal_prob)
        smooth_prob = sum(self.cnn_prob_buffer) / len(self.cnn_prob_buffer)
        return smooth_prob > 0.77

    def analyze_frame(self, frame, fps: float) -> AttentionFrameResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_for_inference = (
            cv2.resize(
                rgb,
                None,
                fx=self.process_scale,
                fy=self.process_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            if 0 < self.process_scale < 1.0
            else rgb
        )

        face_results = self.face_mesh.process(rgb_for_inference)
        pose_results = self.pose.process(rgb)

        face_detected = bool(face_results.multi_face_landmarks)
        pose_angles = PoseAngles()
        gaze_direction = "Unknown"
        blink_bpm = 0
        eye_focus_score = 0.0
        eye_status_msg = "Face Not Detected (Eye disabled)"
        current_face_landmarks = None
        upper_body_state = UpperBodyState()
        is_drowsy = False

        if pose_results.pose_landmarks:
            upper_body_state = self.upper_body_analyzer.estimate(
                pose_results.pose_landmarks,
                frame.shape[1],
                frame.shape[0],
            )

        if face_results.multi_face_landmarks:
            current_face_landmarks = face_results.multi_face_landmarks[0]
            pose_angles = self._update_head_pose_state(current_face_landmarks, frame)

            eye_result = self.eye_focus_analyzer.analyze(frame, current_face_landmarks)
            gaze_direction = eye_result.gaze_direction
            blink_bpm = eye_result.blink_bpm
            eye_focus_score = eye_result.eye_focus_score
            eye_status_msg = eye_result.eye_status_msg

            cnn_frame = frame.copy()

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            mp_drawing.draw_landmarks(
                image=cnn_frame,
                landmark_list=current_face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

            if self.refine_landmarks:
                mp_drawing.draw_landmarks(
                    image=cnn_frame,
                    landmark_list=current_face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

            is_drowsy = self._check_cnn_state(cnn_frame, current_face_landmarks)
        else:
            current_face_landmarks = None
            pose_angles = PoseAngles()
            self.eye_focus_analyzer.reset()
            gaze_direction = "Unknown"
            blink_bpm = 0
            eye_focus_score = 0.0
            eye_status_msg = "Face Not Detected (Eye disabled)"

        dt = 1.0 / max(fps, 1e-6)
        try:
            attention_state = self.attention_analyzer.update(
                pose_angles=pose_angles,
                body_tilt=upper_body_state.shoulder_tilt,
                face_detected=face_detected,
                dt=dt,
                gaze_direction=gaze_direction,
                blink_bpm=blink_bpm,
                eye_focus_score=eye_focus_score,
                eye_status_msg=eye_status_msg,
                is_drowsy=is_drowsy,
            )
        except TypeError:
            attention_state = self.attention_analyzer.update(
                pose_angles=pose_angles,
                body_tilt=upper_body_state.shoulder_tilt,
                face_detected=face_detected,
                dt=dt,
                gaze_direction=gaze_direction,
                blink_bpm=blink_bpm,
                eye_focus_score=eye_focus_score,
                eye_status_msg=eye_status_msg,
            )

        return AttentionFrameResult(
            face_detected=face_detected,
            pose_angles=pose_angles,
            gaze_direction=gaze_direction,
            blink_bpm=blink_bpm,
            eye_focus_score=eye_focus_score,
            eye_status_msg=eye_status_msg,
            upper_body_state=upper_body_state,
            attention_state=attention_state,
            is_drowsy=is_drowsy,
            current_face_landmarks=current_face_landmarks,
        )

    def close(self) -> None:
        self.face_mesh.close()
        self.pose.close()