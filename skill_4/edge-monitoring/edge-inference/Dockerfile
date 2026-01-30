FROM python:3.10-slim

WORKDIR /app
RUN pip install onnxruntime numpy websockets

COPY infer.py .
COPY model.onnx .

CMD ["python", "infer.py"]
