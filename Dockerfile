FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_fastapi.py .
COPY segmentation_engine.py .
COPY onnx/ ./onnx/

ENV PORT=8080

CMD exec uvicorn app_fastapi:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 60
