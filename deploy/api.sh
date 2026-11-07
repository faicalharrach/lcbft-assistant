#!/usr/bin/env bash
# Construit et déploie l'API de criblage sur Cloud Run (accès authentifié uniquement).
set -euo pipefail

PROJET=$(gcloud config get-value project)
REGION=europe-west1
SA="lcbft-app@${PROJET}.iam.gserviceaccount.com"

gcloud run deploy lcbft-api \
  --source=api \
  --region="$REGION" \
  --service-account="$SA" \
  --no-allow-unauthenticated \
  --memory=512Mi \
  --min-instances=0 \
  --max-instances=2

echo "URL : $(gcloud run services describe lcbft-api --region="$REGION" --format='value(status.url)')"