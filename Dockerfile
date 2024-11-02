FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    git \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/app/h5_models/VGG16_model.h5
ENV MODEL_PATH=/app/app/P

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
