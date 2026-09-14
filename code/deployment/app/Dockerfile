# Stage 3b - image for the Streamlit application.
#
# The app carries no model and no scientific stack: it is a thin HTTP client,
# which is why its image is an order of magnitude smaller than the API's.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY code/deployment/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/app.py ./app.py

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"]

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
