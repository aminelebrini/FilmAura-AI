
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow

WORKDIR /opt/filmaura
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements-airflow.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}
COPY . .

ENV PYTHONPATH=/opt/filmaura
