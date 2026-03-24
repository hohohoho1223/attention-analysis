# run_ml.py
import os
from ml.run_preprocess_ml import run_extract_vids
from ml.train_ml import train, MODE

if __name__ == '__main__':
    # os.makedirs(f'./ml/trained_models_{MODE}', exist_ok=True)
    run_extract_vids()
    train()