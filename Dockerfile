FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ari ./ari
COPY knowledge ./knowledge
COPY evals/dataset.jsonl ./evals/dataset.jsonl

EXPOSE 8000
CMD ["python", "-m", "ari", "serve", "--host", "0.0.0.0", "--port", "8000"]
