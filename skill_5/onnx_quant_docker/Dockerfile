# Step 1: Base image with Python
FROM python:3.9-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install required packages
RUN pip install --no-cache-dir \
    torch \
    onnx \
    onnxruntime \
    onnxruntime-tools \
    numpy

# Step 4: Copy project files
COPY generate_fp32_model.py .
COPY quantize_model.py .

# Step 5: Default command
CMD ["bash"]
