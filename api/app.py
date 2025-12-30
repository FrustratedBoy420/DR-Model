from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Model Load karein (Path sahi rakhiyega)
MODEL_PATH = os.path.join(os.getcwd(), "amd_model.keras")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.get("/")
def home():
    return {"status": "AMD Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Image read karein
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    # 2. Preprocessing (Jo aapne Gradio mein ki thi)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Prediction
    prediction = model.predict(img_array)
    pred_value = float(prediction[0][0])

    if pred_value >= 0.5:
        result = f"AMD Detected ({pred_value:.2f})"
    else:
        result = f"Normal ({1 - pred_value:.2f})"

    return result