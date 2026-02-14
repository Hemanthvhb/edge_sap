from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import time

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    
    # 1. Load and Preprocess Image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    
    # Normalize to 0-1
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # (Optional) Add ImageNet Mean/Std normalization here for higher accuracy
    
    # Reshape: (H, W, C) -> (C, H, W) and add Batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2. Run Inference
    outputs = session.run([output_name], {input_name: img_array})
    raw_output = outputs[0][0]
    
    # 3. Process Results
    probabilities = softmax(raw_output)
    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    latency = time.time() - start
    
    return {
        "prediction_class": prediction,
        "confidence": confidence,
        "inference_time_sec": round(latency, 4)
    }