# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app



# Install Python deps
RUN pip install --no-cache-dir tensorflow-cpu psutil numpy

# Copy code
COPY app.py .

# Run the script
CMD ["python", "app.py"]
