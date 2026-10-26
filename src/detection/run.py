"""Lance la détection de bout en bout : features -> scores -> évaluation -> sauvegarde.

Usage :
    python -m src.detection.run                 # lit et écrit dans BigQuery
    python -m src.detection.run --source local  # tout en local (DuckDB + CSV)

Sorties :
- data/scores/scores_clients.csv  : score, rang, règles déclenchées, raisons SHAP ;
- data/scores/shap_clients.csv    : contribution de chaque variable, par client ;
- reports/evaluation.md           : métriques contre la vérité terrain ;
- BigQuery lcbft_features.scores_clients (mode bigquery).
"""

import argparse

import pandas as pd

from src.data import charger_features, charger_labels, client_bigquery
from src.detection.evaluate import preparer_evaluation, rapport, sauver_rapport
from src.detection.model import scorer
from src.generate.config import PROJECT_ROOT, RAW_DIR

SCORES_DIR = PROJECT_ROOT / "data" / "scores"


def sauver_local(scores: pd.DataFrame, shap_df: pd.DataFrame) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    a_plat = scores.assign(
        regles_declenchees=scores["regles_declenchees"].str.join("|"),
        raisons_modele=scores["raisons_modele"].str.join("|"),
    )
    a_plat.to_csv(SCORES_DIR / "scores_clients.csv", index=False)
    shap_df.assign(numero_dossier=scores["numero_dossier"]).to_csv(SCORES_DIR / "shap_clients.csv", index=False)


def sauver_bigquery(scores: pd.DataFrame) -> str:
    from google.cloud import bigquery

    client = client_bigquery()
    table_id = f"{client.project}.lcbft_features.scores_clients"
    S = bigquery.SchemaField
    config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            S("numero_dossier", "STRING", "REQUIRED"),
            S("score_if", "FLOAT64"),
            S("score_regles", "FLOAT64"),
            S("score_final", "FLOAT64"),
            S("regles_declenchees", "STRING", "REPEATED"),
            S("raisons_modele", "STRING", "REPEATED"),
            S("rang", "INT64"),
        ],
    )
    client.load_table_from_dataframe(scores, table_id, job_config=config).result()
    return table_id


def main(source: str) -> None:
    print(f"Source : {source}")
    features = charger_features(source)
    print(f"{len(features):,} clients chargés, calcul des scores et de SHAP...")
    scores, shap_df = scorer(features)

    sauver_local(scores, shap_df)
    if source == "bigquery":
        print(f"Scores écrits dans {sauver_bigquery(scores)}")

    profils_csv = RAW_DIR / "profils_legitimes.csv"
    profils = pd.read_csv(profils_csv) if profils_csv.exists() else None
    evaluation = preparer_evaluation(scores, charger_labels(source), profils)
    texte = rapport(evaluation)
    print("\n" + texte)
    print(f"Rapport : {sauver_rapport(texte)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["bigquery", "local"], default="bigquery")
    main(parser.parse_args().source)
