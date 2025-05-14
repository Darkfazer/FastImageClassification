from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

app = FastAPI()

# Enable CORS so frontend can access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Image prediction logic
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        predictions = predict_image(temp_file)
        results = [{"class": p[1], "confidence": float(p[2])} for p in predictions]
        return {"predictions": results}
    finally:
        os.remove(temp_file)