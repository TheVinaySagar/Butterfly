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

RUN python -c "import os; assert os.path.exists('app/Custom_CNN_Model.h5'), 'Model file not found'"

ENV MODEL_PATH=/app/app/Custom_CNN_Model.h5

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
