FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU), ONNX, ONNX Runtime
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision \
    torchaudio \
    onnx \
    onnxruntime \
    numpy

COPY prune_and_export.py .

CMD ["python", "prune_and_export.py"]