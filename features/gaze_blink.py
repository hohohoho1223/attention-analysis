import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time

class FocusAnalyzer:
    def __init__(self):
        # MediaPipe 설정
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # 홍채(Iris) 좌표 추출을 위해 필수
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 눈 관련 랜드마크 인덱스 (MediaPipe 기준)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [473, 474, 475, 476]
        self.RIGHT_IRIS = [468, 469, 470, 471]

        # 데이터 스무딩을 위한 버퍼 (최근 5프레임)
        self.ear_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=5)
        
        # 캘리브레이션 및 상태 관리
        self.is_calibrated = False
        self.calibration_frames = 0
        self.MAX_CALIB_FRAMES = 100
        self.base_ear = 0.0
        
        # 눈 깜빡임 및 집중도 카운터
        self.blink_count = 0
        self.eye_closed = False
        self.last_gaze_move_time = time.time()
        self.gaze_variance_buffer = deque(maxlen=30) # 멍 때리기 감지용 시선 변화량 저장

    def _calculate_ear(self, eye_points):
        """Eye Aspect Ratio(EAR) 계산: 눈의 세로/가로 비율"""
        try:
            # 수직 거리 계산
            v1 = dist.euclidean(eye_points[1], eye_points[5])
            v2 = dist.euclidean(eye_points[2], eye_points[4])
            # 수평 거리 계산
            h = dist.euclidean(eye_points[0], eye_points[3])
            ear = (v1 + v2) / (2.0 * h)
            return ear
        except Exception:
            return None

    def _get_gaze_ratio(self, eye_points, iris_points):
        try:
            # 양쪽 눈 끝점의 x좌표를 정렬하여 무조건 화면상 왼쪽(x_min)과 오른쪽(x_max)을 구분
            eye_corners_x = sorted([eye_points[0][0], eye_points[3][0]])
            eye_x_min, eye_x_max = eye_corners_x[0], eye_corners_x[1]

            # 홍채 중심점의 x좌표
            iris_center_x = np.mean(iris_points, axis=0)[0]

            # 눈의 가로 폭
            total_width = eye_x_max - eye_x_min
            if total_width <= 0: return 0.5, 0

            # 홍채가 왼쪽 끝에서부터 떨어진 거리 비율
            ratio = (iris_center_x - eye_x_min) / total_width
            return ratio, total_width
        except Exception:
            return 0.5, 0

    # 통합 : 외부에서 faceMesh를 받기 위해 내부에서 faceMesh를 계산하는 코드는 주석처리
    # def analyze(self, frame):
    def analyze(self, face_landmarks, w, h): # frame 대신 face_landmarks 받음
        """메인 분석 함수"""
        # h, w, _ = frame.shape
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = self.face_mesh.process(rgb_frame)
        
        status = {"blink_rate": self.blink_count, "gaze": "Center", "focus_score": 100, "msg": "Analyzing...", "gaze_ratio": None, "eye_openness": None} # ML용 feature 추가

        # if not results.multi_face_landmarks:
        #     status["msg"] = "Face Not Detected (Check Background/Blur)"
        #     return status

        # # 첫 번째 얼굴 데이터 추출
        # face_landmarks = results.multi_face_landmarks[0]
        
        # 좌표 변환 (Normalized -> Pixel)
        def get_coords(indices):
            return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

        left_eye_coords = get_coords(self.LEFT_EYE)
        right_eye_coords = get_coords(self.RIGHT_EYE)
        left_iris_coords = get_coords(self.LEFT_IRIS)

        right_iris_coords = get_coords(self.RIGHT_IRIS)

        # 1. 시선 추적 (Gaze) - '더 잘 보이는 눈' 동적 선택
        ratio_l, width_l = self._get_gaze_ratio(left_eye_coords, left_iris_coords)
        ratio_r, width_r = self._get_gaze_ratio(right_eye_coords, right_iris_coords)

        # 가로 폭이 더 긴(카메라를 향해 왜곡이 적은) 눈의 비율을 사용
        current_gaze_ratio = ratio_l if width_l > width_r else ratio_r

        self.gaze_buffer.append(current_gaze_ratio)
        smooth_gaze = sum(self.gaze_buffer) / len(self.gaze_buffer)

        # 건우님 눈동자 움직임에 맞춘 임계값 적용
        if smooth_gaze < 0.48: status["gaze"] = "Left"
        elif smooth_gaze > 0.58: status["gaze"] = "Right"
        else: status["gaze"] = "Center"

        status["gaze_ratio"] = smooth_gaze # ML용 feature 값 추가
        
        # 2. EAR 계산 및 노이즈 필터링
        ear_l = self._calculate_ear(left_eye_coords)
        ear_r = self._calculate_ear(right_eye_coords)
        
        if ear_l and ear_r:
            avg_ear = (ear_l + ear_r) / 2.0
            self.ear_buffer.append(avg_ear)
            smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            
            # 3. 캘리브레이션 (초기화 단계)
            if not self.is_calibrated:
                self.calibration_frames += 1
                self.base_ear += smooth_ear
                status["msg"] = f"Calibrating... {self.calibration_frames}%"
                status["eye_openness"] = smooth_ear # ML용 feature 값 추가; 정지된 이미지용 (눈 깜빡임 추출이 불가능할 때)
                if self.calibration_frames >= self.MAX_CALIB_FRAMES:
                    self.base_ear /= self.MAX_CALIB_FRAMES
                    self.is_calibrated = True
                return status

            # 4. 눈 깜빡임 감지 (임계값: 개인 평균의 70% 미만일 때)
            blink_threshold = self.base_ear * 0.75
            if smooth_ear < blink_threshold:
                if not self.eye_closed:
                    self.blink_count += 1
                    self.eye_closed = True
            else:
                self.eye_closed = False

            # 5. '멍 때리기' 감지 로직 스케치
            # 시선 변화량 계산
            self.gaze_variance_buffer.append(smooth_gaze)
            gaze_variance = np.var(self.gaze_variance_buffer) if len(self.gaze_variance_buffer) > 20 else 1.0
            
            # 로직: 시선 이동이 거의 없고(variance < 0.001) 눈을 거의 안 깜빡이면 멍 때림 의심
            # 실제 운영 시에는 특정 시간(예: 10초) 동안의 누적 지표로 판단
            if gaze_variance < 0.0005 and not self.eye_closed:
                status["msg"] = "Spacing Out Detected (Low Gaze Activity)"
                status["focus_score"] = 40
            else:
                status["msg"] = "Studying..."
                status["focus_score"] = 90

        return status
