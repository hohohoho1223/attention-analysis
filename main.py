import pandas as pd
from features.run_feature_extract import run_feature_extract

if __name__ == "__main__":

    # feature 추출 (최초 1회만 실행, 이후 주석처리)
    # run_feature_extract()

    # 저장된 feature 불러오기
    df = pd.read_csv('./features.csv')
    
    # 모델 로드