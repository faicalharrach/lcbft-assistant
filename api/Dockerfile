FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Puis le code, dans un dossier "api" pour que les imports "from api.xxx" fonctionnent
COPY . /app/api

# Cloud Run fournit le port dans la variable PORT (8080 par défaut)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]