# inference_viewer.py
# 실시간 육안 확인용 — 파일 선택 다이얼로그로 영상 열기

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import joblib
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from features.head_pose import HeadPoseEstimator
from features.gaze_blink import FocusAnalyzer

# ── 설정 ──────────────────────────────────────────
COORD_DIR  = "./data/Zoom_Record/crop_coords"
MODEL_PATH = "./ml/trained_models/model_xgb_3.joblib"
SCALER_PATH = "./ml/trained_models/scaler.joblib"
# ─────────────────────────────────────────────────

# 라벨별 색상 (BGR)
LABEL_COLOR = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 0)}
LABEL_TEXT  = {0: "UnFocus", 1: "Partial", 2: "Focus"}

# 파일 선택 다이얼로그
tk.Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="영상 선택",
    filetypes=[("MP4 파일", "*.mp4")]
)
if not video_path:
    print("영상이 선택되지 않았어요.")
    exit()

video_path = Path(video_path)
coord_path = Path(COORD_DIR) / f"{video_path.stem}.json"

if not coord_path.exists():
    print(f"[ERROR] 좌표 파일 없음: {coord_path}")
    exit()

# 좌표 로드
with open(coord_path) as f:
    coords = json.load(f)["students"]

# 모델, 스케일러 로드
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# mediapipe 초기화 (학생 5명 각각 독립적인 analyzer 필요)
mp_face_mesh = mp.solutions.face_mesh
face_meshes = [
    mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True
    ) for _ in range(len(coords))
]
head_pose_estimators = [HeadPoseEstimator() for _ in range(len(coords))]
gaze_blink_analyzers = [FocusAnalyzer()    for _ in range(len(coords))]


def extract_feature(frame, face_mesh, head_pose_estimator, gaze_blink_analyzer):
    """단일 프레임에서 feature 추출"""
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    lm_list = []
    pitch, yaw, roll = None, None, None
    gaze_ratio, eye_openness = None, None
    ih, iw, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크 추출
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * iw), int(lm.y * ih)
                lm_list += [x, y]

            # head pose
            head_pose = head_pose_estimator.estimate(face_landmarks, iw, ih)
            if head_pose is not None:
                pitch, yaw, roll = head_pose.pitch, head_pose.yaw, head_pose.roll

            # gaze & eye_openness
            eyes_result = gaze_blink_analyzer.analyze(face_landmarks, iw, ih)
            gaze_ratio   = eyes_result.get("gaze_ratio")
            eye_openness = eyes_result.get("eye_openness")  # 캘리브레이션 전에만 값 있음

    if len(lm_list) == 0:
        return None

    row = {}
    for i in range(468):
        row[f'x{i}'] = lm_list[i*2]     if i*2   < len(lm_list) else -1
        row[f'y{i}'] = lm_list[i*2 + 1] if i*2+1 < len(lm_list) else -1
    row['pitch']        = pitch
    row['yaw']          = yaw
    row['roll']         = roll
    row['gaze_ratio']   = gaze_ratio
    row['eye_openness'] = eye_openness  # None이면 NaN으로 자동 처리됨
    row['blink'] = None

    return row


def predict(row):
    """feature 딕셔너리 → 집중도 라벨 예측"""
    df = pd.DataFrame([row])
    df = df.astype(float)  # None → NaN 전체 변환 (ML은 None을 처리하지 못함)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    df_scaled = df_scaled.drop(columns=['blink'], errors='ignore') # 임시 (단일 프레임이라 blink 없음)

    # 판정 편파
    proba = model.predict_proba(df_scaled)[0]  # [unfocus, partial, focus]
    
    if proba[0] >= 0.5: # unfocus 확률이 x 미만이면 partial/focus 중에서 판정
        label = 0  # unfocus
    elif proba[1] >= 0.5: # partial 확률이 x 미만이면 focus로 판정
        label = 1  # partial
    else:
        label = 2  # focus
        
    return label


# ── 메인 루프 ──────────────────────────────────────
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) * 2 if fps > 0 else 33 * 2

print(f"[{video_path.name}] 재생 시작 — q: 종료, d: 앞으로 10초, a: 뒤로 10초")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    labels = []

    for i, c in enumerate(coords):
        # 학생별 독립적인 analyzer 사용
        crop = frame[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']]

        row = extract_feature(
            crop,
            face_meshes[i],
            head_pose_estimators[i],
            gaze_blink_analyzers[i]
        )

        if row is not None:
            label = predict(row)
        else:
            label = -1  # 얼굴 인식 실패

        print(f"S{i+1} row: {row is None}, label: {label}")  # 추가 # 이후제거

        labels.append(label)

        # 크롭 영역에 라벨 + 색상 오버레이
        color = LABEL_COLOR.get(label, (128, 128, 128))
        text  = LABEL_TEXT.get(label, "No face")
        cv2.rectangle(frame,
                      (c['x'], c['y']),
                      (c['x']+c['w'], c['y']+c['h']),
                      color, 2)
        cv2.putText(frame, f"S{i+1} {text}",
                    (c['x']+4, c['y']+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 집중도 총합 표시
    count = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
    cv2.putText(frame, f"focus:{count[2]}  partial:{count[1]}  Unfocus:{count[0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Inference Viewer", frame)

    key = cv2.waitKey(delay) & 0xFF  # 1ms만 대기
    if key == ord('q'):
        break
    elif key == ord('d'):  # d: 앞으로 10초
        current = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, current + 10000)
    elif key == ord('a'):  # a: 뒤로 10초
        current = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, current - 10000)

cap.release()
cv2.destroyAllWindows()