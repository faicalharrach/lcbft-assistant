FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY . .

# Données synthétiques et scores générés à la construction de l'image.
# Les graines aléatoires étant fixes, ce sont les mêmes données qu'en local.
RUN python -m src.generate.run_all && python -m src.detection.run --source local

# Données lues dans l'image, journal des décisions dans BigQuery, agent sur Vertex AI.
ENV APP_SOURCE=local \
    JOURNAL=bigquery \
    AGENT_LLM=vertex \
    RAG_MOTEUR=vertex

EXPOSE 8080
CMD streamlit run src/app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
