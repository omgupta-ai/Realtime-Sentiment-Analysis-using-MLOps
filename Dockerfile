# File: streaming/Dockerfile (Updated)
FROM python:3.9-slim
WORKDIR /app

# Correct path to requirements file
COPY streaming/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training.1600000.processed.noemoticon.csv .

# Correct path to copy the scripts (producer.py, consumer.py)
COPY streaming/ .