import h5py
import json


with h5py.File("face_defense_model.h5", "r") as f:
    print("keras_version:", f.attrs.get("keras_version"))
    print("backend:", f.attrs.get("backend"))
    print("attrs:", list(f.attrs.keys()))

    model_config = f.attrs.get("model_config")
    if model_config:
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        config_json = json.loads(model_config)
        print(config_json.keys())
        print(config_json["class_name"])