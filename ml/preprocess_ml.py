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

from head_pose import HeadPoseEstimator
from eye_focus import EyeFocusAnalyzer


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
WINDOW_SIZE=30

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

    # 결과 초기값
    pitch_buf, yaw_buf, roll_buf = [],[],[]
    gaze_buf, blink_buf = [],[]
    window_stats = [] # X 프레임 통계값 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_mesh.process(frame)
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
            pitch_buf.append(pitch)
            yaw_buf.append(yaw)
            roll_buf.append(roll)
            gaze_buf.append(gaze_ratio)
            blink_buf.append(blink)
        
        if len(pitch_buf) == WINDOW_SIZE: # 버퍼가 X프레임 찼는지 확인
            row = {
                'vid_id': vid_id, 'window_id': window_idx,
                'pitch_mean': np.mean(pitch_buf), 'pitch_std': np.std(pitch_buf),
                'yaw_mean': np.mean(yaw_buf),     'yaw_std': np.std(yaw_buf),
                'roll_mean': np.mean(roll_buf),   'roll_std': np.std(roll_buf),
                'gaze_mean': np.mean(gaze_buf),   'gaze_std': np.std(gaze_buf),
                'blink_mean': np.mean(blink_buf),
            }
            window_stats.append(row)

            window_idx += 1

            # 버퍼 비우기
            pitch_buf, yaw_buf, roll_buf = [], [], []
            gaze_buf, blink_buf = [], []

    cap.release()
    return pd.DataFrame(window_stats)


def _build_feature_data(landmark, label, include_blink):
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



