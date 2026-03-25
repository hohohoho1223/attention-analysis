import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import time

class FocusAnalyzer:
    def __init__(self):
        # 1. MediaPipe 설정 (다시 원래대로 깔끔하게!)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. MobileNetV2 설정 (방어막 엔진) 🌟 [추가됨]
        print("방어막 모델을 로드하는 중...")
        try:
            self.defense_model = tf.keras.models.load_model('face_defense_model.h5') # 같은 폴더에 모델 파일 필수!
            print("✅ 방어막 모델 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            exit()
            
        self.class_names = ['0_normal', '1_glare', '2_occlusion']
        
        # 눈 관련 랜드마크 인덱스
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [473, 474, 475, 476]
        self.RIGHT_IRIS = [468, 469, 470, 471]

        # 데이터 스무딩을 위한 버퍼
        self.ear_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=15)
        
        # 캘리브레이션 및 상태 관리
        self.is_calibrated = False
        self.calibration_frames = 0
        self.MAX_CALIB_FRAMES = 100
        self.base_ear = 0.0
        
        # 눈 깜빡임(BPM) 및 멍때림 감지용 변수
        self.blink_timestamps = deque() 
        self.eye_closed = False
        self.gaze_variance_buffer = deque(maxlen=30)
        self.start_time = time.time()
        
        # 🌟 [추가됨] 예외 상황 발생 시 직전 점수를 유지하기 위한 변수
        self.last_score = 100  

    def _check_edge_case(self, frame, face_landmarks, w, h, padding=15):
        """🌟 [완벽 수정됨] 에러 없이 얼굴 전체(Full Face)를 224x224로 판별합니다."""
        try:
            # 1. 눈만 자르는 낡은 로직은 삭제! 대신 웹캠 전체 화면(frame)을 사용합니다.
            # 모델이 학습했던 224x224 사이즈로 맞춰줍니다.
            img_resized = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 2. 모델에 넣기 위한 차원 확장 및 전처리
            img_array = np.expand_dims(img_rgb, axis=0)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.astype(np.float32))
            
            # 3. 예측 수행
            predictions = self.defense_model.predict(img_preprocessed, verbose=0)
            
            # 4. 캐글 모델은 클래스가 딱 2개! (0: 정상, 1: 가려짐)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # 5. 가려짐(인덱스 1)이라고 80% 이상 확신할 때만 Paused 발동!
            if class_idx == 1 and confidence > 0.8: 
                return "2_occlusion" # 아래 메인 로직과 호환성을 위해 2_occlusion으로 문자열 반환
            else:
                return "0_normal"
                
        except Exception as e:
            # 이제 에러가 나면 무조건 가리지 않고, 터미널에 이유를 알려줍니다!
            print(f"🚨 방어막 에러 발생: {e}")
            return "0_normal" # 에러 시 일단 정상으로 넘겨서 시스템 멈춤 방지

    def _calculate_ear(self, eye_points):
        """Eye Aspect Ratio(EAR) 계산"""
        try:
            v1 = dist.euclidean(eye_points[1], eye_points[5])
            v2 = dist.euclidean(eye_points[2], eye_points[4])
            h = dist.euclidean(eye_points[0], eye_points[3])
            return (v1 + v2) / (2.0 * h)
        except:
            return None

    def _get_gaze_ratio(self, eye_points, iris_points):
        """시선 방향 계산"""
        try:
            eye_corners_x = sorted([eye_points[0][0], eye_points[3][0]])
            eye_x_min, eye_x_max = eye_corners_x[0], eye_corners_x[1]
            iris_center_x = np.mean(iris_points, axis=0)[0]

            total_width = eye_x_max - eye_x_min
            if total_width <= 0: return 0.5, 0

            return (iris_center_x - eye_x_min) / total_width, total_width
        except:
            return 0.5, 0

    def analyze(self, frame):
        """메인 분석 함수"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        status = {"bpm": len(self.blink_timestamps), "gaze": "Center", "focus_score": self.last_score, "msg": "Analyzing...", "edge_case": False}

        if not results.multi_face_landmarks:
            status["msg"] = "Face Not Detected"
            status["focus_score"] = 0
            self.last_score = 0
            return status

        face_landmarks = results.multi_face_landmarks[0]

        # 🌟 1. 방어막 가동 (Edge Case 검사)
        eye_state = self._check_edge_case(frame, face_landmarks, w, h)
        
        if eye_state != '0_normal':
            # 악조건(빛반사, 가려짐) 감지 시: EAR 및 Gaze 계산을 '스킵'하고 직전 상태 유지
            status["edge_case"] = True
            if eye_state == '1_glare':
                status["msg"] = "Shield Active: Glare (Paused)"
            else:
                status["msg"] = "Shield Active: Occlusion (Paused)"
            return status # 계산하지 않고 바로 리턴!

        # 🌟 2. 정상 상태일 때만 아래 MediaPipe 로직(EAR, Gaze) 수행
        def get_coords(indices):
            return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

        left_eye_coords = get_coords(self.LEFT_EYE)
        right_eye_coords = get_coords(self.RIGHT_EYE)
        left_iris_coords = get_coords(self.LEFT_IRIS)
        right_iris_coords = get_coords(self.RIGHT_IRIS)
        
        ear_l = self._calculate_ear(left_eye_coords)
        ear_r = self._calculate_ear(right_eye_coords)
        
        if ear_l and ear_r:
            avg_ear = (ear_l + ear_r) / 2.0
            self.ear_buffer.append(avg_ear)
            smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            
            if not self.is_calibrated:
                self.calibration_frames += 1
                self.base_ear += smooth_ear
                status["msg"] = f"Calibrating... {self.calibration_frames}%"
                if self.calibration_frames >= self.MAX_CALIB_FRAMES:
                    self.base_ear /= self.MAX_CALIB_FRAMES
                    self.is_calibrated = True
                return status

            # BPM 계산
            current_time = time.time()
            blink_threshold = self.base_ear * 0.75
            
            if smooth_ear < blink_threshold:
                if not self.eye_closed:
                    self.blink_timestamps.append(current_time)
                    self.eye_closed = True
            else:
                self.eye_closed = False

            while self.blink_timestamps and current_time - self.blink_timestamps[0] > 60:
                self.blink_timestamps.popleft()
            
            current_bpm = len(self.blink_timestamps)
            status["bpm"] = current_bpm

            # 시선 추적
            ratio_l, width_l = self._get_gaze_ratio(left_eye_coords, left_iris_coords)
            ratio_r, width_r = self._get_gaze_ratio(right_eye_coords, right_iris_coords)
            current_gaze_ratio = ratio_l if width_l > width_r else ratio_r

            self.gaze_buffer.append(current_gaze_ratio)
            smooth_gaze = sum(self.gaze_buffer) / len(self.gaze_buffer)

            if smooth_gaze < 0.44: status["gaze"] = "Left"
            elif smooth_gaze > 0.62: status["gaze"] = "Right"
            else: status["gaze"] = "Center"

            # 집중도 판별
            self.gaze_variance_buffer.append(smooth_gaze)
            gaze_variance = np.var(self.gaze_variance_buffer) if len(self.gaze_variance_buffer) > 20 else 1.0
            elapsed_time = current_time - self.start_time

            if status["gaze"] != "Center":
                status["msg"] = "Distracted (Looking Away)"
                status["focus_score"] = 40
            elif current_bpm > 15:
                status["msg"] = "Anxious/Distracted (High BPM)"
                status["focus_score"] = 60
            elif elapsed_time > 10 and current_bpm < 3 and gaze_variance < 0.0005:
                status["msg"] = "Spacing Out (Low BPM & Fixed Gaze)"
                status["focus_score"] = 50
            else:
                status["msg"] = "Focused (Optimal)"
                status["focus_score"] = 100

            self.last_score = status["focus_score"] # 정상 점수 업데이트

        return status

# --- 테스트 실행부 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    analyzer = FocusAnalyzer()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1) # 좌우 반전
        res = analyzer.analyze(frame)
        
        # 🌟 UI 표시 분기 처리 (방어막 작동 시 UI 색상 변경)
        msg_color = (0, 255, 255) if res.get('edge_case') else ((0, 0, 255) if res['focus_score'] < 80 else (0, 255, 0))
        
        cv2.putText(frame, f"Gaze: {res['gaze']}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if res['gaze'] == "Center" else (0, 0, 255), 2)
        cv2.putText(frame, f"BPM: {res['bpm']}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if 3 <= res['bpm'] <= 15 else (0, 165, 255), 2)
        cv2.putText(frame, f"Score: {res['focus_score']}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {res['msg']}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

        cv2.imshow('Education Focus Monitor', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()