"""Accès aux données : BigQuery (par défaut) ou local (DuckDB sur data/raw/).

Le mode local exécute le même SQL BigQuery, traduit vers DuckDB par sqlglot.
Il sert aux tests et à travailler hors ligne.
"""

import os

import pandas as pd

from src.generate.config import PROJECT_ROOT, RAW_DIR

SQL_FEATURES = PROJECT_ROOT / "sql" / "features_clients.sql"
TABLES_RAW = ["ref_mcc", "ref_pays", "ref_marchands", "clients", "transactions"]


# --------------------------------------------------------------------------
# Local
# --------------------------------------------------------------------------
def connexion_locale():
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE SCHEMA raw")
    for table in TABLES_RAW:
        con.execute(f"CREATE TABLE raw.{table} AS SELECT * FROM read_csv_auto('{RAW_DIR / table}.csv')")
    return con


def executer_local(sql_bigquery: str, con=None) -> pd.DataFrame:
    import sqlglot

    con = con or connexion_locale()
    sql_duckdb = sqlglot.transpile(sql_bigquery, read="bigquery", write="duckdb")[0]
    return con.execute(sql_duckdb).df()


# --------------------------------------------------------------------------
# BigQuery
# --------------------------------------------------------------------------
def client_bigquery():
    from google.cloud import bigquery

    from src.bigquery.load_raw import LOCATION

    return bigquery.Client(project=os.environ.get("GCP_PROJECT_ID"), location=LOCATION)


# --------------------------------------------------------------------------
# Points d'entrée
# --------------------------------------------------------------------------
def charger_features(source: str = "bigquery") -> pd.DataFrame:
    if source == "local":
        from src.generate.transactions import SEUIL_ALERTE_ESPECES

        sql = SQL_FEATURES.read_text().format(raw="raw", seuil_especes=SEUIL_ALERTE_ESPECES)
        return executer_local(sql)
    client = client_bigquery()
    return client.query(
        f"SELECT * FROM `{client.project}`.lcbft_features.features_clients"
    ).to_dataframe()


def charger_labels(source: str = "bigquery") -> pd.DataFrame:
    if source == "local":
        return pd.read_csv(RAW_DIR / "labels_clients.csv")
    client = client_bigquery()
    return client.query(
        f"SELECT * FROM `{client.project}`.lcbft_labels.labels_clients"
    ).to_dataframe()
