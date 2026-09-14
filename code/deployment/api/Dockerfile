# Stage 3a - image for the model API.
#
# Build context is the repository root (see code/deployment/docker-compose.yml)
# because the image needs two things that live outside this folder:
#   * models/            - the artefact produced by Stage 2
#   * code/models/features.py - the shared feature contract
#
# Layer order is deliberate: requirements change rarely and are installed
# first, so the 5-minute rebuild triggered by a fresh model only re-runs the
# cheap COPY layers below and finishes in a couple of seconds.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY code/deployment/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Shared feature engineering - identical to the one used during training.
COPY code/models/features.py ./features.py
COPY code/deployment/api/main.py ./main.py

# The trained model. This layer is invalidated on every pipeline run, which is
# exactly the intent: a new model means a new image.
COPY models/model.pkl models/model_card.json models/metrics.json ./models/

EXPOSE 8000

# python:slim ships no curl, so the healthcheck is written in Python. The exec
# form is used deliberately: it bypasses the shell entirely, so nothing in the
# quoting can be reinterpreted differently by another builder.
HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
