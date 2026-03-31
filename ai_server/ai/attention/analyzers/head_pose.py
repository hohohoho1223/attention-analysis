from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 263,
    "right_eye_outer": 33,
    "left_mouth": 291,
    "right_mouth": 61,
}


@dataclass
class PoseAngles:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class HeadPoseEstimator:
    def __init__(self) -> None:
        self.model_points = {
            "nose_tip": (0.0, 0.0, 0.0),
            "chin": (0.0, -63.6, -12.5),
            "left_eye_outer": (-43.3, 32.7, -26.0),
            "right_eye_outer": (43.3, 32.7, -26.0),
            "left_mouth": (-28.9, -28.9, -24.1),
            "right_mouth": (28.9, -28.9, -24.1),
        }
    
    def _build_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray: # 카메라 내부 파라미터 행렬 생성
        focal_length = float(frame_width)
        center = (frame_width / 2.0, frame_height / 2.0)
        return np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _build_dist_coeffs(self) -> np.ndarray: # 왜곡 계수 생성
        return np.zeros((4, 1), dtype=np.float64)

    def _get_image_points( # MediaPipe 랜드마크 → 2D 좌표 변환
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        image_points = []
        for key in HEAD_POSE_LANDMARKS:
            idx = HEAD_POSE_LANDMARKS[key]
            landmark = face_landmarks.landmark[idx]
            image_points.append(
                [landmark.x * frame_width, landmark.y * frame_height]
            )
        return np.array(image_points, dtype=np.float64)

    def _get_object_points(self) -> np.ndarray: # 기준 3D 얼굴 점 생성
        object_points = [self.model_points[key] for key in HEAD_POSE_LANDMARKS]
        return np.array(object_points, dtype=np.float64)

    def _compute_sy(self, rotation_matrix: np.ndarray) -> float: # singular 판정용 값 계산
        return math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    def _is_singular(self, sy: float) -> bool: # singular 여부 판정
        return sy < 1e-6
    
    # 각 3개의 축별 계산 메서드
    def _calculate_pitch(self, rotation_matrix: np.ndarray, singular: bool) -> float:
        if not singular:
            return math.degrees(
                math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])
            )
        return math.degrees(
            math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        )

    def _calculate_yaw(self, rotation_matrix: np.ndarray, sy: float) -> float:
        return math.degrees(math.atan2(rotation_matrix[2, 0], sy))

    def _calculate_roll(self, rotation_matrix: np.ndarray, singular: bool) -> float:
        if not singular:
            return math.degrees(
                math.atan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
            )
        return 0.0
    
    # 세 각도를 PoseAngles로 묶음
    def _rotation_matrix_to_angles(self, rotation_matrix: np.ndarray) -> PoseAngles:
        sy = self._compute_sy(rotation_matrix)
        singular = self._is_singular(sy)

        pitch = self._calculate_pitch(rotation_matrix, singular)
        yaw = self._calculate_yaw(rotation_matrix, sy)
        roll = self._calculate_roll(rotation_matrix, singular)

        return PoseAngles(yaw=yaw, pitch=pitch, roll=roll)

    def _solve_pnp( # PnP 계산
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ):
        return cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    
    def _rotation_vector_to_matrix(self, rotation_vector: np.ndarray) -> np.ndarray: # 회전 벡터 → 회전 행렬
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return rotation_matrix
    
    # PnP 계산 → 회전 행렬 → Euler 각도 계산의 전체 흐름을 담당하는 메서드
    # PnP 결과를 최종 각도로 변환
    def _estimate_pose_angles(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[PoseAngles]:
        success, rotation_vector, _translation_vector = self._solve_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
        )
        if not success:
            return None

        rotation_matrix = self._rotation_vector_to_matrix(rotation_vector)
        return self._rotation_matrix_to_angles(rotation_matrix)

    # 최종적으로 외부에서 호출되는 메서드로, MediaPipe 랜드마크와 프레임 크기를 받아서 PoseAngles를 반환
    def estimate(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Optional[PoseAngles]:
        image_points = self._get_image_points(
            face_landmarks,
            frame_width,
            frame_height,
        )
        object_points = self._get_object_points()
        camera_matrix = self._build_camera_matrix(frame_width, frame_height)
        dist_coeffs = self._build_dist_coeffs()

        return self._estimate_pose_angles(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
        )