import os
# 🌟 [가장 중요] 최신 텐서플로우에게 옛날(Keras 2) 엔진을 사용하라고 강제 명령! (무조건 맨 위에 있어야 합니다)
# ai_serveㅇr/main.py에서 설정한 것과 동일하게 환경변수를 설정하여 TensorFlow가 레거시 Keras 엔진을 사용하도록 강제합니다. 이렇게 하면 모델 호환성 문제를 방지할 수 있습니다.
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import base64
import io
from collections import deque
from pathlib import Path
from typing import Any, Dict

import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from ai.attention.analyzers.attention_logic import (
    AttentionAnalyzer,
    AttentionConfig,
    DEFAULT_ATTENTION_CONFIG,
)
from ai.attention.analyzers.eye_focus import EyeFocusAnalyzer
from ai.attention.analyzers.head_pose import HeadPoseEstimator, PoseAngles
from ai.attention.analyzers.upperbody_pose import UpperBodyAnalyzer, UpperBodyState

try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError(
        'TensorFlow import 에 실패했습니다. 현재 환경의 tensorflow / keras 버전 충돌 가능성이 큽니다.'
    ) from exc

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / 'face_defense_model.h5'

SHARED_DEFENSE_MODEL = None

def load_shared_defense_model(enable_defense_model: bool = True):
    global SHARED_DEFENSE_MODEL

    model_name = 'face_defense_model.h5'
    model_path = MODEL_PATH

    if not enable_defense_model:
        print('[INFO] CNN 방어막 모델 비활성화 상태로 실행합니다.')
        return None
    if SHARED_DEFENSE_MODEL is not None:
        return SHARED_DEFENSE_MODEL
    if not model_path.exists():
        print(f"[WARN] '{model_name}' 파일을 찾을 수 없어 CNN 방어막 없이 실행합니다.")
        return None

    print(f'🛡️ AI 방어막 모델({model_name})을 로드하는 중...')
    try:
        try:
            SHARED_DEFENSE_MODEL = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            SHARED_DEFENSE_MODEL = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False,
            )
        print('✅ 1티어 방어막 모델 로드 완료!')
    except Exception as e:
        SHARED_DEFENSE_MODEL = None
        print(f'❌ 모델 로드 실패. CNN 방어막 없이 계속 실행합니다: {e}')

    return SHARED_DEFENSE_MODEL

app = FastAPI(title='CV Focus AI Server')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

class PredictionRequest(BaseModel):
    left_eye: str
    right_eye: str
    mouth: str

class AnalyzeFrameRequest(BaseModel):
    frame: str
    session_id: str = 'default'
    process_scale: float = 0.75
    refine_landmarks: bool = True
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    fps: float = 30.0
    enable_defense_model: bool = True

class FrameAnalyzerSession:
    def __init__(
        self,
        *,
        process_scale: float = 0.75,
        refine_landmarks: bool = True,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        enable_defense_model: bool = True,
        attention_config: AttentionConfig = DEFAULT_ATTENTION_CONFIG,
    ) -> None:
        self.process_scale = process_scale
        self.refine_landmarks = refine_landmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.enable_defense_model = enable_defense_model
        self.attention_config = attention_config

        self.pose_angles = PoseAngles()
        self.face_detected = False
        self.gaze_direction = 'Unknown'
        self.blink_bpm = 0
        self.eye_focus_score = 100.0
        self.eye_status_msg = 'Eye analysis disabled'
        self.current_face_landmarks = None
        self.upper_body_state = UpperBodyState()

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self.defense_model = None
        self.tf = tf
        self.cnn_prob_buffer = deque(maxlen=8)
        self._load_defense_model()

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.head_pose_estimator = HeadPoseEstimator()
        self.upper_body_analyzer = UpperBodyAnalyzer()
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.attention_analyzer = AttentionAnalyzer(config=self.attention_config)
        self.eye_focus_analyzer = EyeFocusAnalyzer()
    
    def _load_defense_model(self) -> None:
        self.defense_model = load_shared_defense_model(self.enable_defense_model)
        if self.defense_model is None:
            self.tf = None


    def _update_head_pose_state(self, face_landmarks, frame) -> None:
        frame_height, frame_width = frame.shape[:2]
        pose = self.head_pose_estimator.estimate(face_landmarks, frame_width, frame_height)
        if pose is not None:
            self.pose_angles = pose

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

    def _check_cnn_state(self, frame, face_landmarks):
        if self.defense_model is None:
            return False, {
                'left_eye': 0.0,
                'right_eye': 0.0,
                'mouth': 0.0,
                'smooth': 0.0,
            }

        h, w = frame.shape[:2]
        mouth_indices = [61, 291, 39, 181, 0, 17, 269, 405]

        left_eye = self._get_crop_img(frame, face_landmarks, self.LEFT_EYE, w, h, padding=15)
        right_eye = self._get_crop_img(frame, face_landmarks, self.RIGHT_EYE, w, h, padding=15)
        mouth = self._get_crop_img(frame, face_landmarks, mouth_indices, w, h, padding=60)

        imgs = [img for img in [left_eye, right_eye, mouth] if img is not None]
        if not imgs:
            return False, {
                'left_eye': 0.0,
                'right_eye': 0.0,
                'mouth': 0.0,
                'smooth': 0.0,
            }

        img_array = np.array(imgs).astype(np.float32)
        predictions = self.defense_model.predict(img_array, verbose=0)

        if len(predictions) == 3:
            print(
                f"[CNN 분석] 좌안: {predictions[0][1] * 100:.1f}% | "
                f"우안: {predictions[1][1] * 100:.1f}% | "
                f"입(하관): {predictions[2][1] * 100:.1f}%"
            )

        p_left = float(predictions[0][1]) if len(predictions) > 0 else 0.0
        p_right = float(predictions[1][1]) if len(predictions) > 1 else 0.0
        p_mouth = float(predictions[2][1]) if len(predictions) > 2 else 0.0
        max_abnormal_prob = float(np.max(predictions[:, 1]))

        self.cnn_prob_buffer.append(max_abnormal_prob)
        smooth_prob = sum(self.cnn_prob_buffer) / len(self.cnn_prob_buffer)

        return smooth_prob > 0.77, {
            'left_eye': p_left,
            'right_eye': p_right,
            'mouth': p_mouth,
            'smooth': smooth_prob,
        }

    def draw_mediapipe(self, frame):
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

        results = self.face_mesh.process(rgb_for_inference)
        self.face_detected = bool(results.multi_face_landmarks)

        pose_results = self.pose.process(rgb)
        if pose_results.pose_landmarks:
            self.upper_body_state = self.upper_body_analyzer.estimate(
                pose_results.pose_landmarks,
                frame.shape[1],
                frame.shape[0],
            )
        else:
            self.upper_body_state = UpperBodyState()

        if results.multi_face_landmarks:
            self.current_face_landmarks = results.multi_face_landmarks[0]
            for face_landmarks in results.multi_face_landmarks:
                self._update_head_pose_state(face_landmarks, frame)
                eye_result = self.eye_focus_analyzer.analyze(frame, face_landmarks)
                self.gaze_direction = eye_result.gaze_direction
                self.blink_bpm = eye_result.blink_bpm
                self.eye_focus_score = eye_result.eye_focus_score
                self.eye_status_msg = eye_result.eye_status_msg

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
        else:
            self.current_face_landmarks = None
            self.pose_angles = PoseAngles()
            self.face_detected = False
            self.eye_focus_analyzer.reset()
            self.gaze_direction = 'Unknown'
            self.blink_bpm = 0
            self.eye_focus_score = 0.0
            self.eye_status_msg = 'Face Not Detected (Eye disabled)'

        return frame

    def analyze_frame(self, frame_bgr: np.ndarray, fps: float = 30.0) -> dict:
        output = self.draw_mediapipe(frame_bgr)

        cnn_probs = {
            'left_eye': 0.0,
            'right_eye': 0.0,
            'mouth': 0.0,
            'smooth': 0.0,
        }
        is_drowsy = False
        if self.face_detected and self.current_face_landmarks:
            is_drowsy, cnn_probs = self._check_cnn_state(output, self.current_face_landmarks)

        dt = 1.0 / max(fps, 1e-6)
        attention_state = self.attention_analyzer.update(
            pose_angles=self.pose_angles,
            body_tilt=self.upper_body_state.shoulder_tilt,
            face_detected=self.face_detected,
            dt=dt,
            gaze_direction=self.gaze_direction,
            blink_bpm=self.blink_bpm,
            eye_focus_score=self.eye_focus_score,
            eye_status_msg=self.eye_status_msg,
            is_drowsy=is_drowsy,
        )
        
        # FastAPI가 JSON으로 변환해서 HTTP 응답으로 보냄
        return {
            'state': attention_state.state,
            'score': float(attention_state.score),
            'face_detected': bool(self.face_detected),
            'gaze_direction': self.gaze_direction,
            'is_fixated': bool(attention_state.is_fixated),
            'blink_bpm': int(self.blink_bpm),
            'eye_focus_score': float(self.eye_focus_score),
            'eye_status_msg': self.eye_status_msg,
            'pose': {
                'yaw': float(self.pose_angles.yaw),
                'pitch': float(self.pose_angles.pitch),
                'roll': float(self.pose_angles.roll),
            },
            'body': {
                'visible': bool(self.upper_body_state.body_visible),
                'shoulder_tilt': float(self.upper_body_state.shoulder_tilt),
            },
            'durations': {
                'head': float(attention_state.head_duration),
                'body': float(attention_state.body_duration),
                'fixation_break': float(attention_state.fixation_break_duration),
                'no_face': float(attention_state.no_face_duration),
            },
            'cnn': {
                'is_drowsy': bool(is_drowsy),
                **cnn_probs,
            },
        }


sessions: Dict[str, FrameAnalyzerSession] = {}


def get_session(request: AnalyzeFrameRequest) -> FrameAnalyzerSession:
    session = sessions.get(request.session_id)
    if session is None:
        session = FrameAnalyzerSession(
            process_scale=request.process_scale,
            refine_landmarks=request.refine_landmarks,
            detection_confidence=request.detection_confidence,
            tracking_confidence=request.tracking_confidence,
            enable_defense_model=request.enable_defense_model,
        )
        sessions[request.session_id] = session
    return session


def decode_base64_image(b64_str: str) -> np.ndarray:
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def decode_crop_image(b64_str: str) -> np.ndarray:
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    return np.array(img, dtype=np.float32) / 255.0


@app.get('/health')
async def health() -> dict:
    return {'status': 'ok'}


# @app.post('/predict')
# async def predict(req: PredictionRequest) -> dict:
#     defense_model = load_shared_defense_model(True)
#     if defense_model is None:
#         raise HTTPException(status_code=500, detail='CNN 방어막 모델이 로드되지 않았습니다.')

#     left = decode_crop_image(req.left_eye)
#     right = decode_crop_image(req.right_eye)
#     mouth = decode_crop_image(req.mouth)

#     batch = np.stack([left, right, mouth])
#     preds = defense_model.predict(batch, verbose=0)

#     p_left = float(preds[0][1])
#     p_right = float(preds[1][1])
#     p_mouth = float(preds[2][1])
#     max_prob = max(p_left, p_right, p_mouth)

#     return {
#         'prob': max_prob,
#         'left_eye': p_left,
#         'right_eye': p_right,
#         'mouth': p_mouth,
#     }


@app.post('/predict')
async def predict(payload: Dict[str, Any]) -> dict:
    # 1) main_before 전체 분석 경로: frame 1장 전달
    if 'frame' in payload:
        req = AnalyzeFrameRequest(**payload)
        try:
            frame_bgr = decode_base64_image(req.frame)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'프레임 디코딩 실패: {exc}') from exc

        session = get_session(req)
        try:
            return session.analyze_frame(frame_bgr, fps=req.fps)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'프레임 분석 실패: {exc}') from exc

    # 2) 구형 CNN 확률 추론 경로: left_eye/right_eye/mouth 3장 전달
    if all(k in payload for k in ('left_eye', 'right_eye', 'mouth')):
        req = PredictionRequest(**payload)
        defense_model = load_shared_defense_model(True)
        if defense_model is None:
            raise HTTPException(status_code=500, detail='CNN 방어막 모델이 로드되지 않았습니다.')

        left = decode_crop_image(req.left_eye)
        right = decode_crop_image(req.right_eye)
        mouth = decode_crop_image(req.mouth)

        batch = np.stack([left, right, mouth])
        preds = defense_model.predict(batch, verbose=0)

        p_left = float(preds[0][1])
        p_right = float(preds[1][1])
        p_mouth = float(preds[2][1])
        max_prob = max(p_left, p_right, p_mouth)

        return {
            'prob': max_prob,
            'left_eye': p_left,
            'right_eye': p_right,
            'mouth': p_mouth,
        }

    raise HTTPException(
        status_code=422,
        detail='요청 본문 형식이 올바르지 않습니다. frame 또는 left_eye/right_eye/mouth 가 필요합니다.',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CV Focus AI Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    return parser


if __name__ == '__main__':
    import uvicorn

    args = build_parser().parse_args()
    uvicorn.run('main:app', host=args.host, port=args.port, reload=args.reload)