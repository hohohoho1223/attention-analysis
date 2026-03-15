from features.feature_extract import faceMesh_extract, buildFeatureDataframe
import pandas as pd

def run_feature_extract():
    # 경로 정의
    path1 = './WACV_data/dataset/1'
    path2 = './WACV_data/dataset/2'
    path3 = './WACV_data/dataset/3'

    # feature 추출
    lm_dic0 = faceMesh_extract(path1)
    df0 = buildFeatureDataframe(lm_dic0, 0)
    print(f"폴더1: {len(df0)}행")

    lm_dic1 = faceMesh_extract(path2)
    df1 = buildFeatureDataframe(lm_dic1, 1)
    print(f"폴더2: {len(df1)}행")

    lm_dic2 = faceMesh_extract(path3)
    df2 = buildFeatureDataframe(lm_dic2, 2)
    print(f"폴더3: {len(df2)}행")

    # 합치고 저장
    df_all = pd.concat([df0, df1, df2], ignore_index=True)
    df_all.to_csv('./features.csv', index=False)
    print(f"저장 완료: {len(df_all)}행")