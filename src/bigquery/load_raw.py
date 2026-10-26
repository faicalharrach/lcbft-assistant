"""Charge les tables brutes (data/raw/*.csv) dans BigQuery.

Deux datasets séparés :
- lcbft_raw    : les données que le modèle a le droit de voir ;
- lcbft_labels : la vérité terrain, utilisée seulement pour l'évaluation.
Les séparer évite qu'une requête de features lise les étiquettes par erreur.

Choix BigQuery :
- schémas explicites (pas d'autodetect) : types maîtrisés, NUMERIC pour l'argent ;
- transactions partitionnée par jour sur date_trans et clusterisée par
  numero_dossier : les requêtes par période ou par client lisent moins de données.

Prérequis : gcloud auth application-default login
Projet : variable d'environnement GCP_PROJECT_ID, sinon le projet gcloud par défaut.
"""

import os

from google.cloud import bigquery

from src.generate.config import RAW_DIR

LOCATION = "EU"
DATASET_RAW = "lcbft_raw"
DATASET_LABELS = "lcbft_labels"

S = bigquery.SchemaField

# L'ordre des colonnes doit être celui des CSV : BigQuery charge les CSV par position.
TABLES = {
    "ref_mcc": (DATASET_RAW, [
        S("mcc", "INT64", "REQUIRED"),
        S("libelle", "STRING"),
        S("famille", "STRING"),
        S("niveau_risque", "STRING"),
    ]),
    "ref_pays": (DATASET_RAW, [
        S("code_pays", "STRING", "REQUIRED"),
        S("libelle", "STRING"),
        S("niveau_risque", "STRING"),
    ]),
    "ref_marchands": (DATASET_RAW, [
        S("marchand_id", "STRING", "REQUIRED"),
        S("nom", "STRING"),
        S("mcc", "INT64"),
        S("code_pays", "STRING"),
    ]),
    "clients": (DATASET_RAW, [
        S("numero_dossier", "STRING", "REQUIRED"),
        S("numero_foyer", "STRING"),
        S("nom", "STRING"),
        S("prenom", "STRING"),
        S("age", "INT64"),
        S("nationalite", "STRING"),
        S("ccsp", "STRING"),
        S("salaire_declare", "NUMERIC"),
        S("pension", "NUMERIC"),
        S("revenu_foyer", "NUMERIC"),
        S("pays_residence", "STRING"),
        S("date_ouverture", "DATE"),
    ]),
    "transactions": (DATASET_RAW, [
        S("transaction_id", "STRING", "REQUIRED"),
        S("numero_dossier", "STRING", "REQUIRED"),
        S("date_trans", "TIMESTAMP", "REQUIRED"),
        S("montant", "NUMERIC", "REQUIRED"),
        S("sens", "STRING"),
        S("type_trans", "STRING"),
        S("marchand_id", "STRING"),
        S("code_pays", "STRING"),
    ]),
    "labels_clients": (DATASET_LABELS, [
        S("numero_dossier", "STRING", "REQUIRED"),
        S("typologie", "STRING"),
        S("intensite", "STRING"),
    ]),
    "labels_transactions": (DATASET_LABELS, [
        S("transaction_id", "STRING", "REQUIRED"),
        S("numero_dossier", "STRING"),
        S("typologie", "STRING"),
    ]),
    "profils_legitimes": (DATASET_LABELS, [
        S("numero_dossier", "STRING", "REQUIRED"),
        S("profil", "STRING"),
    ]),
}


def _creer_dataset(client: bigquery.Client, nom: str) -> None:
    dataset = bigquery.Dataset(f"{client.project}.{nom}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)


def _job_config(table: str, schema) -> bigquery.LoadJobConfig:
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    if table == "transactions":
        config.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date_trans"
        )
        config.clustering_fields = ["numero_dossier"]
    return config


def charger(client: bigquery.Client) -> None:
    for dataset in {d for d, _ in TABLES.values()}:
        _creer_dataset(client, dataset)

    for table, (dataset, schema) in TABLES.items():
        chemin = RAW_DIR / f"{table}.csv"
        if not chemin.exists():
            raise FileNotFoundError(f"{chemin} absent : lance d'abord python -m src.generate.run_all")
        table_id = f"{client.project}.{dataset}.{table}"
        with open(chemin, "rb") as f:
            job = client.load_table_from_file(f, table_id, job_config=_job_config(table, schema))
        job.result()  # attend la fin du chargement et lève une erreur s'il échoue
        nb = client.get_table(table_id).num_rows
        print(f"{table_id:<50} {nb:>9,} lignes")


if __name__ == "__main__":
    client = bigquery.Client(project=os.environ.get("GCP_PROJECT_ID"), location=LOCATION)
    print(f"Projet : {client.project}\n")
    charger(client)
