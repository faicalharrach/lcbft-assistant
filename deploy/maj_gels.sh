#!/usr/bin/env bash
# Déploie la Cloud Function maj-gels et la planifie chaque nuit à 3 h (heure de Paris).
set -euo pipefail

PROJET=$(gcloud config get-value project)
REGION=europe-west1
SA="lcbft-app@${PROJET}.iam.gserviceaccount.com"

echo "1/4 Activation des API"
gcloud services enable cloudfunctions.googleapis.com cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com

echo "2/4 Déploiement de la fonction"
gcloud functions deploy maj-gels \
  --gen2 \
  --runtime=python312 \
  --region="$REGION" \
  --source=functions/maj_gels \
  --entry-point=maj_gels \
  --trigger-http \
  --no-allow-unauthenticated \
  --service-account="$SA" \
  --memory=512Mi \
  --timeout=300s

URL=$(gcloud functions describe maj-gels --gen2 --region="$REGION" --format='value(serviceConfig.uri)')

echo "3/4 Droit d'appel pour le compte de service"
gcloud functions add-invoker-policy-binding maj-gels --region="$REGION" --member="serviceAccount:${SA}"

echo "4/4 Planification chaque nuit à 3 h"
if gcloud scheduler jobs describe maj-gels-nuit --location="$REGION" >/dev/null 2>&1; then
  ACTION=update
else
  ACTION=create
fi
gcloud scheduler jobs "$ACTION" http maj-gels-nuit \
  --location="$REGION" \
  --schedule="0 3 * * *" \
  --time-zone="Europe/Paris" \
  --uri="$URL" \
  --http-method=POST \
  --oidc-service-account-email="$SA" \
  --oidc-token-audience="$URL"

echo "Terminé. Fonction : $URL"