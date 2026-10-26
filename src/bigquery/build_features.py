"""Construit la table lcbft_features.features_clients dans BigQuery.

La requête est dans sql/features_clients.sql (un SELECT pur, testable en local
avec tests/sql_local.py). Ce script la lance dans BigQuery et écrit le résultat
dans une table de destination, remplacée à chaque exécution.
"""

import os
from pathlib import Path

from google.cloud import bigquery

from src.bigquery.load_raw import DATASET_RAW, LOCATION, _creer_dataset
from src.generate.config import PROJECT_ROOT
from src.generate.transactions import SEUIL_ALERTE_ESPECES

DATASET_FEATURES = "lcbft_features"
SQL_PATH = PROJECT_ROOT / "sql" / "features_clients.sql"


def requete(projet: str) -> str:
    raw = f"`{projet}`.{DATASET_RAW}"
    return Path(SQL_PATH).read_text().format(raw=raw, seuil_especes=SEUIL_ALERTE_ESPECES)


def construire(client: bigquery.Client) -> str:
    _creer_dataset(client, DATASET_FEATURES)
    destination = f"{client.project}.{DATASET_FEATURES}.features_clients"
    config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        clustering_fields=["numero_dossier"],
    )
    job = client.query(requete(client.project), job_config=config)
    job.result()
    table = client.get_table(destination)
    mo = (job.total_bytes_processed or 0) / 1e6
    print(f"{destination} : {table.num_rows:,} lignes, {len(table.schema)} colonnes")
    print(f"Données lues : {mo:.1f} Mo")
    return destination


if __name__ == "__main__":
    client = bigquery.Client(project=os.environ.get("GCP_PROJECT_ID"), location=LOCATION)
    construire(client)
