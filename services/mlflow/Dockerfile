# Read-only MLflow UI over the file store written by Stage 2.
# Built from python:slim rather than pulled from a registry so the version is
# pinned to the same 2.17.2 that mlflow-skinny writes with.
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow==2.17.2"
EXPOSE 5000
