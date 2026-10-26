"""Journal des décisions de l'analyste : traçabilité de chaque alerte traitée.

Chaque décision est AJOUTÉE (jamais modifiée ni supprimée) avec sa date, son auteur,
sa justification et le score du moment. Le statut d'un dossier est sa dernière décision.
C'est ce qu'attend un contrôle : pouvoir reconstituer qui a décidé quoi, quand et pourquoi
(conservation des résultats de l'analyse, art. L. 561-12 du Code monétaire et financier).

Stockage :
- JOURNAL=bigquery : table lcbft_features.decisions (déploiement, données partagées) ;
- JOURNAL=local    : data/decisions/decisions.csv (poste de développement).
"""

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.generate.config import PROJECT_ROOT

BACKEND = os.environ.get("JOURNAL", "local")
CHEMIN_LOCAL = PROJECT_ROOT / "data" / "decisions" / "decisions.csv"
TABLE = "lcbft_features.decisions"
COLONNES = ["horodatage", "numero_dossier", "decision", "justification", "analyste",
            "score_final", "note_generee"]


def _client():
    from src.data import client_bigquery

    return client_bigquery()


def _creer_table_si_absente(client) -> str:
    from google.cloud import bigquery

    S = bigquery.SchemaField
    table = bigquery.Table(f"{client.project}.{TABLE}", schema=[
        S("horodatage", "TIMESTAMP", "REQUIRED"),
        S("numero_dossier", "STRING", "REQUIRED"),
        S("decision", "STRING", "REQUIRED"),
        S("justification", "STRING"),
        S("analyste", "STRING"),
        S("score_final", "FLOAT64"),
        S("note_generee", "BOOL"),
    ])
    table.time_partitioning = bigquery.TimePartitioning(field="horodatage")
    table.clustering_fields = ["numero_dossier"]
    client.create_table(table, exists_ok=True)
    return table.table_id


def enregistrer(dossier: str, decision: str, justification: str, analyste: str,
                score_final: float, note_generee: bool) -> None:
    ligne = {
        "horodatage": datetime.now(timezone.utc),
        "numero_dossier": dossier,
        "decision": decision,
        "justification": justification.strip(),
        "analyste": analyste.strip() or "analyste",
        "score_final": float(score_final),
        "note_generee": bool(note_generee),
    }
    if BACKEND == "bigquery":
        from google.cloud import bigquery

        client = _client()
        _creer_table_si_absente(client)
        P = bigquery.ScalarQueryParameter
        client.query(
            f"""INSERT INTO `{client.project}`.{TABLE} ({", ".join(COLONNES)})
                VALUES (@horodatage, @dossier, @decision, @justification, @analyste, @score, @note)""",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                P("horodatage", "TIMESTAMP", ligne["horodatage"]),
                P("dossier", "STRING", dossier),
                P("decision", "STRING", decision),
                P("justification", "STRING", ligne["justification"]),
                P("analyste", "STRING", ligne["analyste"]),
                P("score", "FLOAT64", ligne["score_final"]),
                P("note", "BOOL", ligne["note_generee"]),
            ]),
        ).result()
    else:
        CHEMIN_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([ligne]).to_csv(CHEMIN_LOCAL, mode="a", header=not CHEMIN_LOCAL.exists(), index=False)
    lire.clear()


@st.cache_data(ttl=60, show_spinner=False)
def lire() -> pd.DataFrame:
    """Tout le journal, du plus récent au plus ancien."""
    if BACKEND == "bigquery":
        client = _client()
        _creer_table_si_absente(client)
        journal = client.query(f"SELECT * FROM `{client.project}`.{TABLE}").to_dataframe()
    elif CHEMIN_LOCAL.exists():
        journal = pd.read_csv(CHEMIN_LOCAL)
    else:
        journal = pd.DataFrame(columns=COLONNES)
    journal["horodatage"] = pd.to_datetime(journal["horodatage"], utc=True)
    return journal.sort_values("horodatage", ascending=False).reset_index(drop=True)


def statuts() -> dict:
    """Dernière décision de chaque dossier."""
    journal = lire()
    if journal.empty:
        return {}
    return journal.drop_duplicates("numero_dossier").set_index("numero_dossier")["decision"].to_dict()


def historique(dossier: str) -> pd.DataFrame:
    journal = lire()
    return journal[journal["numero_dossier"] == dossier]
