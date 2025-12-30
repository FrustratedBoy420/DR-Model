from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# Legacy/Versioning issues ke liye
import tf_keras 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ⚠️ Class order exactly wahi jo aapne training ke waqt rakha tha
CLASSES = ["Negative", "Mild Diabetic Retinopathy", "Proliferative Diabetic Retinopathy"]

# Model Load logic
MODEL_PATH = "dr_model_3class_final.keras" # Apni file ka naam yahan sahi rakhiyega
model = None

try:
    # Model ko load karne ki koshish
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception:
    # Agar standard load fail ho toh tf_keras use karein
    model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded using tf_keras wrapper!")

@app.get("/")
def home():
    return {"status": "DR Detection API is Live"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Image Read aur Resize (300x300 for EfficientNet)
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((300, 300))

        # 2. Preprocessing
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # EfficientNet specific preprocessing (Scale/Normalization)
        img_array = preprocess_input(img_array)

        # 3. Model Prediction
        preds = model.predict(img_array)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)

        # Output format fix (String + Number error fix)
        result_text = f"{CLASSES[class_id]}"

        return {"Dettected":result_text,"Chance":round(confidence, 3)}

    except Exception as e:
        return {"error": str(e)}