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


# ==========================================================================
# Notes d'analyse générées par l'agent
# ==========================================================================
# Chaque note est conservée avec ce qui permet de l'auditer : modèle, sources citées,
# résultat des contrôles, nombre de rédactions et tokens consommés. Rouvrir un dossier
# affiche la dernière note sans rappeler Gemini.
TABLE_NOTES = "lcbft_features.notes_ia"
CHEMIN_NOTES = PROJECT_ROOT / "data" / "decisions" / "notes_ia.jsonl"


def _creer_table_notes(client) -> None:
    from google.cloud import bigquery

    S = bigquery.SchemaField
    table = bigquery.Table(f"{client.project}.{TABLE_NOTES}", schema=[
        S("horodatage", "TIMESTAMP", "REQUIRED"),
        S("numero_dossier", "STRING", "REQUIRED"),
        S("modele", "STRING"),
        S("note", "STRING"),
        S("sources", "STRING", "REPEATED"),
        S("problemes", "STRING", "REPEATED"),
        S("tentatives", "INT64"),
        S("tokens_entree", "INT64"),
        S("tokens_sortie", "INT64"),
    ])
    table.time_partitioning = bigquery.TimePartitioning(field="horodatage")
    table.clustering_fields = ["numero_dossier"]
    client.create_table(table, exists_ok=True)


def enregistrer_note(dossier: str, modele: str, resultat: dict) -> None:
    brouillons = resultat.get("brouillons", [])
    ligne = {
        "horodatage": datetime.now(timezone.utc),
        "numero_dossier": dossier,
        "modele": modele,
        "note": resultat["note"],
        "sources": [d.id for d in resultat["documents"]],
        "problemes": list(resultat["problemes"]),
        "tentatives": int(resultat["tentatives"]),
        "tokens_entree": int(sum(b.get("usage", {}).get("input_tokens", 0) for b in brouillons)),
        "tokens_sortie": int(sum(b.get("usage", {}).get("output_tokens", 0) for b in brouillons)),
    }
    if BACKEND == "bigquery":
        from google.cloud import bigquery

        client = _client()
        _creer_table_notes(client)
        P, A = bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter
        client.query(
            f"""INSERT INTO `{client.project}`.{TABLE_NOTES}
                (horodatage, numero_dossier, modele, note, sources, problemes, tentatives, tokens_entree, tokens_sortie)
                VALUES (@h, @d, @m, @n, @s, @p, @t, @te, @ts)""",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                P("h", "TIMESTAMP", ligne["horodatage"]), P("d", "STRING", dossier), P("m", "STRING", modele),
                P("n", "STRING", ligne["note"]), A("s", "STRING", ligne["sources"]),
                A("p", "STRING", ligne["problemes"]), P("t", "INT64", ligne["tentatives"]),
                P("te", "INT64", ligne["tokens_entree"]), P("ts", "INT64", ligne["tokens_sortie"]),
            ]),
        ).result()
    else:
        import json

        CHEMIN_NOTES.parent.mkdir(parents=True, exist_ok=True)
        with open(CHEMIN_NOTES, "a", encoding="utf-8") as f:
            f.write(json.dumps(ligne | {"horodatage": ligne["horodatage"].isoformat()}, ensure_ascii=False) + "\n")
    derniere_note.clear()


@st.cache_data(ttl=300, show_spinner=False)
def derniere_note(dossier: str):
    """Dernière note enregistrée pour ce dossier, ou None."""
    if BACKEND == "bigquery":
        from google.cloud import bigquery

        client = _client()
        _creer_table_notes(client)
        lignes = list(client.query(
            f"""SELECT * FROM `{client.project}`.{TABLE_NOTES}
                WHERE numero_dossier = @d ORDER BY horodatage DESC LIMIT 1""",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("d", "STRING", dossier)]),
        ).result())
        return dict(lignes[0]) if lignes else None
    if not CHEMIN_NOTES.exists():
        return None
    import json

    notes = [json.loads(l) for l in CHEMIN_NOTES.read_text(encoding="utf-8").splitlines() if l.strip()]
    notes = [n for n in notes if n["numero_dossier"] == dossier]
    if not notes:
        return None
    note = max(notes, key=lambda n: n["horodatage"])
    note["horodatage"] = pd.to_datetime(note["horodatage"])
    return note
