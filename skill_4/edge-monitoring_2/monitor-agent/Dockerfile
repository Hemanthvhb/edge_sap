FROM python:3.10-slim

WORKDIR /app
RUN pip install psutil websockets

COPY agent.py .
CMD ["python", "agent.py"]

