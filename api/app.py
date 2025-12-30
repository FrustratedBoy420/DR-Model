from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
import tf_keras 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Negative", "Mild Diabetic Retinopathy", "Proliferative Diabetic Retinopathy"]
MODEL_PATH = "dr_model_3class_final.keras"
model = None

# Model Loading logic with better error handling
def load_my_model():
    global model
    try:
        # Pehle standard keras try karein
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"Standard load failed, trying tf_keras: {e}")
        try:
            model = tf_keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded using tf_keras wrapper!")
        except Exception as final_e:
            print(f"❌ Critical Error: Model could not be loaded! {final_e}")

# Start-up par model load karein
load_my_model()

@app.get("/")
def home():
    return {"status": "DR Detection API is Live", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    try:
        # 1. Image Read aur Resize
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # NOTE: Make sure 300x300 matches your training size
        img = img.resize((300, 300))

        # 2. Preprocessing
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # EfficientNet expects specific scaling handled by preprocess_input
        img_array = preprocess_input(img_array)

        # 3. Model Prediction
        preds = model.predict(img_array)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)

        return {
            "Dettected": CLASSES[class_id],
            "Chance": f"{confidence:.2f}%",
            "Status": "Success"
        }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
