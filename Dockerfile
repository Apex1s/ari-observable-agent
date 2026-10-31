FROM python:3.12-slim

# Runs as a non-root user (uid 1000), as required by Hugging Face Spaces.
RUN useradd -m -u 1000 ari
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt tzdata
COPY --chown=ari ari ./ari
COPY --chown=ari knowledge ./knowledge
COPY --chown=ari evals/dataset.jsonl ./evals/dataset.jsonl
RUN mkdir -p /app/data && chown ari /app/data

USER ari
EXPOSE 8000
# Hosting platforms (Render, Koyeb, ...) pass the port to listen on in $PORT.
CMD ["sh", "-c", "python -m ari serve --host 0.0.0.0 --port ${PORT:-8000}"]
