FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/raw/CMaps.dvc ./data/raw/CMaps.dvc
COPY .dvc/config ./.dvc/config

CMD ["python", "src/train_pipeline.py"]