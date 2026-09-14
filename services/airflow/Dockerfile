# Airflow image for the PMLDL pipeline.
#
# Two things are added to the official image:
#   1. the Docker CLI + Compose plugin, so the deployment stage can build and
#      start the API/app containers on the host daemon (the socket is bind
#      mounted in docker-compose.airflow.yml);
#   2. the scientific stack used by Stages 1 and 2.
#
# The scikit-learn / numpy / pandas versions are pinned to exactly the versions
# used in the API image: a model pickled by one version and unpickled by
# another is the single most common way a working pipeline breaks in serving.
FROM apache/airflow:2.10.5-python3.11

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
