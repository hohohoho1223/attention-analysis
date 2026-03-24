# -*- coding: utf-8 -*-
""" File for generating Media Pipe facial features

Original file is located at
    https://colab.research.google.com/drive/1XbJjVsUptvBHMPqyvauLjEBnFyRRz5L4
"""

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from head_pose import HeadPoseEstimator
from eye_focus import EyeFocusAnalyzer

@dataclass
class Buffers:
    lm: list = field(default_factory=list)
    pitch: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    roll: list = field(default_factory=list)
    gaze_ratio: list = field(default_factory=list)
    blink: list = field(default_factory=list)

WINDOW_SIZE=30

# 이미지/단일 프레임 기반
def extract_image_feature(file_path,draw=True):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                             max_num_faces=1,
                                             min_detection_confidence=0.5,
                                             refine_landmarks=True
                                             )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    lm_dic= {}

    # head pose class
    head_pose_estimator = HeadPoseEstimator()
    # gaze, blink class
    eyes_analyzer = EyeFocusAnalyzer()

    for file in sorted(os.listdir(file_path)): # 이후 동영상 전처리 시 000001.jpg, 000002.jpg 처럼 패딩 넣어서 이름 만들기
        image = cv2.imread(os.path.join(file_path,file))
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_images.process(imgRGB)
        lm_list = []

        # 초기값 설정 
        pitch, yaw, roll = None, None, None #얼굴 인식 실패 시 기본값
        gaze_ratio = None #시선 방향
        ih, iw, _ = image.shape #높이 및 너비 기본값

        # 매 프레임마다 버퍼 리셋
        eyes_analyzer.ear_buffer.clear()
        eyes_analyzer.gaze_buffer.clear()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw == True:
                    mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_TESSELATION, mp_drawing_spec, mp_drawing_spec)

                # landmark 추출 (faceMesh)
                for _, lm in enumerate(face_landmarks.landmark):
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    lm_list+=[x,y]

                # head pose
                head_pose = head_pose_estimator.estimate(face_landmarks, iw, ih)
                if head_pose is not None:
                    pitch, yaw, roll = head_pose.pitch, head_pose.yaw, head_pose.roll

                # gaze 
                eyes_result = eyes_analyzer.analyze(image, face_landmarks)
                gaze_ratio = eyes_result.smooth_gaze_ratio

                
        img_id = file.partition('.')[0]
        
        if len(lm_list) == 0:
            lm_dic[img_id] = None # 얼굴 인식이 안 되면 버림
        else:
            lm_dic[img_id] = {
                'landmarks': lm_list,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'gaze_ratio': gaze_ratio,
            }

    return lm_dic

# 비디오 -> 프레임 단위 추출 (이미지 기반 학습 시)
def _preprocess_vid_to_img(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}")
        return
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: # 프레임을 읽었는지 여부
            break
        filename = f"{frame_idx:06d}.jpg" # 000001.jpg 처럼 패딩 넣어서 저장
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        frame_idx += 1

    cap.release()
    print(f"[INFO] 총 {frame_idx}프레임 추출 완료 → {output_dir}")

# X 프레임 windows -> 통계값
def extract_video_feature(video_path, draw=False):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                             max_num_faces=1,
                                             min_detection_confidence=0.5,
                                             refine_landmarks=True
                                             )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # head pose class
    head_pose_estimator = HeadPoseEstimator()
    # gaze, blink class
    eyes_analyzer = EyeFocusAnalyzer()

    cap = cv2.VideoCapture(video_path)

    # 인덱스 값
    vid_id = Path(video_path).stem
    window_idx = 0

    # 초기값
    buffers = Buffers()
    window_stats = []  # X 프레임 통계값 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frameRGB)
        lm_list = []

        # 초기값 설정 
        pitch, yaw, roll = None, None, None #얼굴 인식 실패 시 기본값
        gaze_ratio, blink = None, None #시선 방향
        ih, iw, _ = frame.shape #높이 및 너비 기본값

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw == True:
                    mp_drawing.draw_landmarks(frame, face_landmarks,mp_face_mesh.FACEMESH_TESSELATION, mp_drawing_spec, mp_drawing_spec)

                # landmark 추출 (faceMesh)
                for _, lm in enumerate(face_landmarks.landmark):
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    lm_list+=[x,y]

                # head pose
                head_pose = head_pose_estimator.estimate(face_landmarks, iw, ih)
                if head_pose is not None:
                    pitch, yaw, roll = head_pose.pitch, head_pose.yaw, head_pose.roll

                # gaze 
                eyes_result = eyes_analyzer.analyze(frame, face_landmarks)
                gaze_ratio = eyes_result.smooth_gaze_ratio
                blink = eyes_result.blink_bpm


        # 결과        
        if len(lm_list) != 0: # 얼굴 인식이 될 때만 추가
            buffers.lm.append(lm_list) # # x=468, y=468
            buffers.pitch.append(pitch)
            buffers.yaw.append(yaw)
            buffers.roll.append(roll)
            buffers.gaze_ratio.append(gaze_ratio)
            buffers.blink.append(blink)
        
        if len(buffers.lm) == WINDOW_SIZE: # 버퍼가 X프레임 찼는지 확인
            row = _build_feature_vid(vid_id, window_idx, buffers)
            window_stats.append(row)

            window_idx += 1

            # 버퍼 비우기
            buffers = Buffers()

    cap.release()
    return pd.DataFrame(window_stats)


def _build_feature_img(landmark, label, include_blink):
    rows = []
    for id, data in landmark.items():
        if data is None:
            continue
        row = {"ID": id}
        lm = data['landmarks']
        for i in range(468):
            row[f'x{i}'] = lm[i*2] if i*2 < len(lm) else -1
            row[f'y{i}'] = lm[i*2+1] if i*2+1 < len(lm) else -1
        row['pitch'] = data['pitch']
        row['yaw'] = data['yaw']
        row['roll'] = data['roll']
        row['gaze_ratio'] = data['gaze_ratio']
        if include_blink:
            row['blink'] = data['blink']
        row['Label'] = label
        rows.append(row)
    df = pd.DataFrame(rows)

    return df

def _build_feature_vid(vid_id, window_idx, buffers):
    lm_array = np.array(buffers.lm) # shape -> row : frames, column : coordinates

    # 얼굴 크기 정규화 (카메라와의 거리)
    face_w = np.abs(lm_array[:, 234*2] - lm_array[:, 454*2]) # 얼굴 좌(234)-우(454)의 x 좌표만 계산 
                                                            # lm_array에서 i번째 랜드마크의 x = lm_array[:, i*2], y = lm_array[:, i*2+1] (현재 구조가 index0에 x0, index1에 y1 저장)
    face_h = np.abs(lm_array[:, 10*2+1] - lm_array[:, 152*2+1]) # 얼굴 이마(10) - 턱(152)의 y 좌표만 계산
    face_w = np.where(face_w == 0, 1, face_w) # face_w=0이면 1, 아니면 face_w. 0 나누기 방지
    face_h = np.where(face_h == 0, 1, face_h)

    # 1. 분모가 이상하지 않은지 확인
    if window_idx % 10 == 0: # 너무 많이 찍히니까 10번째 윈도우마다 확인
        print(f"[DEBUG] vid: {vid_id}, win: {window_idx}, face_w_avg: {np.mean(face_w):.2f}")

    row = {'vid_id': vid_id, 'window_id': window_idx}

    # 차분 계산 (익명함수 lambda 사용)
    get_diff_mean = lambda data:np.mean(np.abs(np.diff(data))) if len(data) > 1 else 0
    # 분산 계산
    get_std = lambda data: np.std(data) if len(data) > 1 else 0

    # lm
    for i in range(468): # vid는 len(lm_list) != 0 <- 여기서 0을 거르기 때문에 항상 468 보장
        # lm 정규화
        x_norm = lm_array[:, i*2] / face_w #전체 좌표의 X축 정규화
        y_norm = lm_array[:, i*2+1] / face_h #전체 좌표의 Y축 정규화

        # 2. 정규화된 값의 범위 확인 (너무 작거나 크지 않은지)
        if i == 0 and window_idx == 0: # 첫 번째 랜드마크만 샘플로 확인
            print(f"[DEBUG] x_norm sample (lm 0): {x_norm[:3]}")

        # lm의 프레임 간 움직임 차이
        row[f'x{i}_diff_mean'] = get_diff_mean(x_norm)
        row[f'y{i}_diff_mean']  = get_diff_mean(y_norm)

        row[f'x{i}_std']  = get_std(x_norm)
        row[f'y{i}_std']  = get_std(y_norm)


    # 상체 포즈, 시선 움직임, 눈 깜빡임
    row.update({
        'pitch_diff_mean': get_diff_mean(buffers.pitch), 'pitch_std': get_std(buffers.pitch),
        'yaw_diff_mean': get_diff_mean(buffers.yaw),     'yaw_std': get_std(buffers.yaw),
        'roll_diff_mean': get_diff_mean(buffers.roll),   'roll_std': get_std(buffers.roll),
        'gaze_diff_mean': get_diff_mean(buffers.gaze_ratio),   'gaze_std': get_std(buffers.gaze_ratio),
        'blink_diff_mean': get_diff_mean(buffers.blink),
    })

    return row