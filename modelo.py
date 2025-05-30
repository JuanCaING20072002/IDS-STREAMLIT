import os

model_path = "model/iforest_model_Bot_IoT_pipeline.pkl"
if not os.path.exists(model_path):
    print(f"El archivo {model_path} no existe.")
else:
    print(f"El archivo {model_path} existe. Tamaño: {os.path.getsize(model_path)} bytes")