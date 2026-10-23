# Serving image: numpy + FastAPI over serving/bundle.json. Training/export stay outside (TensorFlow).
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/twotower/__init__.py src/twotower/serve.py src/twotower/
COPY serving/bundle.json serving/
ENV PYTHONPATH=/app/src PORT=8000
USER nobody
EXPOSE 8000
CMD ["sh", "-c", "uvicorn twotower.serve:app --host 0.0.0.0 --port ${PORT}"]
