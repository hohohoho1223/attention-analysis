# ai_server/main.py

import os
# 🌟 [가장 중요] 최신 텐서플로우에게 옛날(Keras 2) 엔진을 사용하라고 강제 명령! (무조건 맨 위에 있어야 합니다)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI(title="CV Focus AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 증강 찌꺼기만 쳐내는 DummyLayer (이건 유지)
class DummyLayer(Layer):
    def __init__(self, *args, **kwargs):
        safe_kwargs = {}
        for k in ['name', 'trainable', 'dtype']:
            if k in kwargs:
                safe_kwargs[k] = kwargs[k]
        super(DummyLayer, self).__init__(**safe_kwargs)

    def call(self, inputs, **kwargs):
        return inputs

custom_objects = {
    'TFOpLambda': DummyLayer,
    'RandomFlip': DummyLayer,
    'RandomRotation': DummyLayer,
    'RandomZoom': DummyLayer,
    'RandomBrightness': DummyLayer,
    'RandomTranslation': DummyLayer,
    'RandomContrast': DummyLayer,
    'Rescaling': DummyLayer
    # 🚨 DepthwiseConv2D 수술 코드는 삭제했습니다! 구버전 엔진이 완벽하게 알아서 읽을 겁니다.
}

print("🔥 [AI 서버] Keras 2 레거시 엔진으로 모델 로딩 중...")
model = tf.keras.models.load_model('face_defense_model.h5', custom_objects=custom_objects, compile=False)
print("✅ [AI 서버] 모델 로드 완벽 성공!")

class PredictionRequest(BaseModel):
    left_eye: str
    right_eye: str
    mouth: str

def decode_image(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

@app.post("/predict")
async def predict(req: PredictionRequest):
    left = decode_image(req.left_eye)
    right = decode_image(req.right_eye)
    mouth = decode_image(req.mouth)

    batch = np.stack([left, right, mouth])
    preds = model.predict(batch, verbose=0)
    
    p_left = float(preds[0][1])
    p_right = float(preds[1][1])
    p_mouth = float(preds[2][1])
    max_prob = max(p_left, p_right, p_mouth)

    return {"prob": max_prob}