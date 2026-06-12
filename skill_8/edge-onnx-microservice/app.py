from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

# Initialize FastAPI app
app = FastAPI(
    title="Edge AI Prediction Service",
    description="ONNX Model Inference Microservice",
    version="1.0.0",
    docs_url="/api-docs",
    redoc_url="/redoc"
)

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Get input & output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Define request body schema
class InputData(BaseModel):
    data: list


@app.get("/")
def read_root():
    return {"message": "Edge AI ONNX Microservice Running"}


@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to numpy array
        input_array = np.array(input_data.data, dtype=np.float32)

        # Add batch dimension if needed
        if len(input_array.shape) == 1:
            input_array = np.expand_dims(input_array, axis=0)

        # Run inference
        result = session.run([output_name], {input_name: input_array})

        return {
            "prediction": result[0].tolist()
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/ui", response_class=HTMLResponse)
def custom_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Edge AI Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--bg); 
                color: var(--text);
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 100vh; 
                margin: 0; 
            }
            .container { 
                background: var(--card-bg); 
                padding: 2.5rem; 
                border-radius: 1rem; 
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                width: 100%;
                max-width: 400px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            h2 { margin-bottom: 1.5rem; font-weight: 600; color: #fff; text-align: center; }
            .input-group { margin-bottom: 1.5rem; }
            label { display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.875rem; }
            input { 
                padding: 0.75rem 1rem; 
                border-radius: 0.5rem; 
                border: 1px solid rgba(255, 255, 255, 0.1);
                background: rgba(0, 0, 0, 0.2);
                color: white;
                width: 100%; 
                box-sizing: border-box; 
                font-size: 1rem;
                transition: all 0.2s;
            }
            input:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
            }
            button { 
                background-color: var(--primary); 
                color: white; 
                border: none; 
                padding: 0.75rem 1rem; 
                border-radius: 0.5rem; 
                cursor: pointer; 
                width: 100%;
                font-size: 1rem;
                font-weight: 600;
                transition: background-color 0.2s;
            }
            button:hover { background-color: var(--primary-hover); }
            .result {
                margin-top: 1.5rem;
                padding: 1rem;
                border-radius: 0.5rem;
                background: rgba(0, 0, 0, 0.2);
                display: none;
            }
            .result pre { margin: 0; color: #10b981; white-space: pre-wrap; word-break: break-all; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Edge AI Predictor</h2>
            <div class="input-group">
                <label for="data">Input Data (comma separated)</label>
                <input id="data" placeholder="e.g. 1, 2, 3, 4" value="1, 2, 3, 4"/>
            </div>
            <button onclick="makePrediction()">Predict</button>
            <div id="result-container" class="result">
                <label>Prediction Result:</label>
                <pre id="result-content"></pre>
            </div>
        </div>

        <script>
            async function makePrediction() {
                const inputVal = document.getElementById('data').value;
                const data = inputVal.split(',').map(item => parseFloat(item.trim())).filter(n => !isNaN(n));
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: data })
                });
                
                const result = await response.json();
                const resultContainer = document.getElementById('result-container');
                const resultContent = document.getElementById('result-content');
                
                resultContainer.style.display = 'block';
                resultContent.textContent = JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    """

