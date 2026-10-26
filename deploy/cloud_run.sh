#!/usr/bin/env bash
# Déploie l'application sur Cloud Run.
# Usage : APP_CODE_ACCES=<code> bash deploy/cloud_run.sh
set -euo pipefail

PROJET="$(gcloud config get-value project 2>/dev/null)"
REGION="${REGION:-europe-west1}"
SERVICE="lcbft-assistant"
COMPTE="lcbft-app"
COMPTE_EMAIL="${COMPTE}@${PROJET}.iam.gserviceaccount.com"
if [ -z "${APP_CODE_ACCES:-}" ]; then
  echo "Définir APP_CODE_ACCES, le code d'accès de la démo. Exemple : APP_CODE_ACCES=monCode bash deploy/cloud_run.sh"
  exit 1
fi

echo "Projet : ${PROJET} · région : ${REGION}"

echo "1/4 Activation des API"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com bigquery.googleapis.com

echo "2/4 Compte de service dédié, avec les seuls droits nécessaires"
gcloud iam service-accounts describe "${COMPTE_EMAIL}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${COMPTE}" --display-name "Application LCB-FT (Cloud Run)"
for ROLE in roles/aiplatform.user roles/bigquery.jobUser roles/bigquery.dataEditor; do
  gcloud projects add-iam-policy-binding "${PROJET}" --member "serviceAccount:${COMPTE_EMAIL}" \
    --role "${ROLE}" --condition=None --quiet >/dev/null
done

echo "3/4 Construction de l'image et déploiement (5 à 10 minutes)"
gcloud run deploy "${SERVICE}" \
  --source . \
  --region "${REGION}" \
  --service-account "${COMPTE_EMAIL}" \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --session-affinity \
  --timeout 300 \
  --set-env-vars "GCP_PROJECT_ID=${PROJET},APP_CODE_ACCES=${APP_CODE_ACCES}"

echo "4/4 Adresse de l'application :"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format "value(status.url)"
