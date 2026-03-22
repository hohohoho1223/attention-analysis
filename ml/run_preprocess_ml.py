import os
import pandas as pd

from ml.preprocess_ml import extract_image_feature, extract_video_feature, _build_feature_data

#이미지 기반 feature 추출
def run_extract_img():
    # 경로 정의
    path1 = './WACV_data/dataset/1'
    path2 = './WACV_data/dataset/2'
    path3 = './WACV_data/dataset/3'

    # feature 추출
    lm_dic0 = extract_image_feature(path1)
    df0 = _build_feature_data(lm_dic0, 0, include_blink=False)
    print(f"폴더1: {len(df0)}행")

    lm_dic1 = extract_image_feature(path2)
    df1 = _build_feature_data(lm_dic1, 1, include_blink=False)
    print(f"폴더2: {len(df1)}행")

    lm_dic2 = extract_image_feature(path3)
    df2 = _build_feature_data(lm_dic2, 2, include_blink=False)
    print(f"폴더3: {len(df2)}행")

    # 합치고 저장
    df_all = pd.concat([df0, df1, df2], ignore_index=True)
    df_all.to_csv('./features.csv', index=False)
    print(f"저장 완료: {len(df_all)}행")

# 비디오 기반 feature 추출
def run_extract_vids():
    # label 0: 이탈, 1: lost-focused, 2: partial-focused, 3: focused
    # 이탈은 ML 학습 후 예외 처리로 테스트 할 예정
    video_dirs = {
        1: './data/my_data/1',
        2: './data/my_data/2',
        3: './data/my_data/3',
    }

    dfs = []
    for label, dir_path in video_dirs.items():
        for video_file in sorted(os.listdir(dir_path)):
            if not video_file.endswith('.mp4'):
                continue
            if video_file.startswith('001'): # 테스트용
                continue
            video_path = os.path.join(dir_path, video_file)
            df = extract_video_feature(video_path)
            df['Label'] = label
            dfs.append(df)
            print(f"{video_file}: {len(df)}행")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv('./features_video.csv', index=False)
    print(f"저장 완료: {len(df_all)}행")