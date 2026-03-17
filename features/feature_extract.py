# -*- coding: utf-8 -*-
""" File for generating Media Pipe facial features

Original file is located at
    https://colab.research.google.com/drive/1XbJjVsUptvBHMPqyvauLjEBnFyRRz5L4
"""

import cv2
import mediapipe as mp
import os
import pandas as pd
from features.head_pose import HeadPoseEstimator
from features.gaze_blink import FocusAnalyzer

# 학습하지 않을 경우 False
wacv = True

if wacv:
    # WACV 데이터셋 학습시
    dataset = "WACV"
else:
    # WACV데이터셋 학습하지 않을 경우
    dataset = "others"

# 랜드마크 추출
def faceMesh_extract(file_path,draw=True):
    g= globals()
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
    gaze_blink_analyzer = FocusAnalyzer()

    for file in sorted(os.listdir(file_path)): # 이후 동영상 전처리 시 000001.jpg, 000002.jpg 처럼 패딩 넣어서 이름 만들기
        image = cv2.imread(os.path.join(file_path,file))
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_images.process(imgRGB)
        lm_list = []

        # 초기값 설정 
        pitch, yaw, roll = None, None, None #얼굴 인식 실패 시 기본값
        gaze_ratio, blink, eye_openness = None, None, None #시선, 눈깜빡임, 눈 감은 비율
        ih, iw, _ = image.shape #높이 및 너비 기본값

        # 매 이미지마다 버퍼 리셋 (WACV는 서로 다른 사람)
        if dataset == "WACV":
            gaze_blink_analyzer.ear_buffer.clear()
            gaze_blink_analyzer.gaze_buffer.clear()

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
                eyes_result = gaze_blink_analyzer.analyze(face_landmarks, iw, ih)
                gaze_ratio = eyes_result.get("gaze_ratio")

                # blink
                if not gaze_blink_analyzer.is_calibrated or dataset=="WACV": # 동일인물이 아니거나 데이터셋이 WACV인 경우
                    eye_openness = eyes_result.get("eye_openness")  # 눈 뜬 비율 값만
                    blink = None
                else: # 연속된 프레임인 경우
                    eye_openness = None
                    blink = eyes_result.get("blink_rate")
                
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
                'eye_openness': eye_openness,
                'blink': blink
            }

    return lm_dic

def buildFeatureDataframe(landmark, label):
    rows = []
    for img_id, data in landmark.items():
        if data is None:
            continue
        row = {"ImageID": img_id}
        lm = data['landmarks']
        for i in range(468):
            row[f'x{i}'] = lm[i*2] if i*2 < len(lm) else -1
            row[f'y{i}'] = lm[i*2+1] if i*2+1 < len(lm) else -1
        row['pitch'] = data['pitch']
        row['yaw'] = data['yaw']
        row['roll'] = data['roll']
        row['gaze_ratio'] = data['gaze_ratio']
        row['eye_openness'] = data['eye_openness']
        row['blink'] = data['blink']
        row['Label'] = label
        rows.append(row)
    df = pd.DataFrame(rows)

    return df

