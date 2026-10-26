"""Chargement des données pour l'interface (mis en cache par Streamlit).

Source par défaut : les fichiers locaux (data/raw et data/scores).
Avec APP_SOURCE=bigquery, les scores et les transactions sont lus dans BigQuery ;
la table transactions étant clusterisée par client, la lecture d'une fiche
client ne parcourt qu'une petite partie de la table.
"""

import os

import pandas as pd
import streamlit as st

from src.detection.model import LIBELLES, REGLES
from src.generate.config import PROJECT_ROOT, RAW_DIR
from src.generate.transactions import SEUIL_ALERTE_ESPECES

SOURCE = os.environ.get("APP_SOURCE", "local")
SCORES_DIR = PROJECT_ROOT / "data" / "scores"

LIBELLES_REGLES = {r.code: r.libelle for r in REGLES}
NOMS_COURTS = {
    "R1_FRACTIONNEMENT": "Fractionnement",
    "R2_COMPTE_DE_PASSAGE": "Compte de passage",
    "R3_PAYS_A_RISQUE": "Pays à risque",
    "R4_INCOHERENCE_REVENUS": "Revenus incohérents",
    "R5_CRYPTO": "Crypto",
    "R6_ESPECES": "Espèces",
    "R7_BIENS_DE_VALEUR": "Biens de valeur",
    "R8_SUR_DEPENSE_CSP": "Sur-dépense CSP",
}
FAMILLES_RISQUE = {
    "jeux": "Jeux d'argent",
    "crypto_change": "Crypto / change",
    "transfert": "Transfert d'argent",
    "prepaye": "Carte prépayée",
    "valeurs": "Biens de valeur",
    "gages": "Prêt sur gages",
}


def libelle_variable(nom: str) -> str:
    return LIBELLES.get(nom, nom.replace("_", " ").capitalize())


def _liste(valeur) -> list:
    if isinstance(valeur, list):
        return valeur
    if hasattr(valeur, "tolist"):
        return list(valeur.tolist())
    if isinstance(valeur, str) and valeur:
        return valeur.split("|")
    return []


@st.cache_data(show_spinner="Chargement des scores...")
def charger_scores() -> pd.DataFrame:
    if SOURCE == "bigquery":
        from src.data import client_bigquery

        client = client_bigquery()
        scores = client.query(
            f"SELECT * FROM `{client.project}`.lcbft_features.scores_clients ORDER BY rang"
        ).to_dataframe()
    else:
        scores = pd.read_csv(SCORES_DIR / "scores_clients.csv")
    scores["regles_declenchees"] = scores["regles_declenchees"].apply(_liste)
    scores["raisons_modele"] = scores["raisons_modele"].apply(_liste)
    clients = charger_clients()
    return scores.merge(clients, on="numero_dossier", how="left").sort_values("rang")


@st.cache_data
def charger_clients() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "clients.csv")


@st.cache_data
def charger_shap() -> pd.DataFrame:
    return pd.read_csv(SCORES_DIR / "shap_clients.csv").set_index("numero_dossier")


@st.cache_data
def charger_referentiels() -> tuple:
    marchands = pd.read_csv(RAW_DIR / "ref_marchands.csv")
    mcc = pd.read_csv(RAW_DIR / "ref_mcc.csv")
    pays = pd.read_csv(RAW_DIR / "ref_pays.csv")
    marchands = marchands.merge(mcc[["mcc", "libelle", "famille"]], on="mcc", how="left")
    return marchands.rename(columns={"nom": "marchand", "libelle": "activite"}), pays


@st.cache_data(show_spinner="Chargement des transactions...")
def _toutes_les_transactions() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date_trans"])


@st.cache_data
def charger_verite_terrain() -> tuple:
    labels = pd.read_csv(RAW_DIR / "labels_clients.csv")
    labels_tx = pd.read_csv(RAW_DIR / "labels_transactions.csv")
    profils_csv = RAW_DIR / "profils_legitimes.csv"
    profils = pd.read_csv(profils_csv) if profils_csv.exists() else pd.DataFrame(columns=["numero_dossier", "profil"])
    return labels, labels_tx, profils


def transactions_client(dossier: str) -> pd.DataFrame:
    if SOURCE == "bigquery":
        from google.cloud import bigquery

        from src.data import client_bigquery

        client = client_bigquery()
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("dossier", "STRING", dossier)
        ])
        tx = client.query(
            f"SELECT * FROM `{client.project}`.lcbft_raw.transactions WHERE numero_dossier = @dossier",
            job_config=config,
        ).to_dataframe()
        tx["montant"] = tx["montant"].astype(float)
    else:
        tx = _toutes_les_transactions()
        tx = tx[tx["numero_dossier"] == dossier]

    marchands, pays = charger_referentiels()
    tx = tx.merge(marchands[["marchand_id", "marchand", "activite", "famille"]], on="marchand_id", how="left")
    tx = tx.merge(pays.rename(columns={"libelle": "pays", "niveau_risque": "risque_pays"}),
                  on="code_pays", how="left")
    tx["signal"] = tx.apply(_signal, axis=1)
    return tx.sort_values("date_trans", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Agrégation des flux par pays...")
def flux_par_pays(dossiers: tuple) -> pd.DataFrame:
    """Flux hors de France des dossiers donnés, agrégés par pays."""
    if SOURCE == "bigquery":
        from google.cloud import bigquery

        from src.data import client_bigquery

        client = client_bigquery()
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("dossiers", "STRING", list(dossiers))
        ])
        agg = client.query(
            f"""SELECT code_pays, SUM(CAST(montant AS FLOAT64)) AS montant,
                       COUNT(*) AS nb_operations, COUNT(DISTINCT numero_dossier) AS nb_clients
                FROM `{client.project}`.lcbft_raw.transactions
                WHERE numero_dossier IN UNNEST(@dossiers) AND code_pays != 'FR'
                GROUP BY code_pays""",
            job_config=config,
        ).to_dataframe()
    else:
        tx = _toutes_les_transactions()
        tx = tx[tx["numero_dossier"].isin(dossiers) & (tx["code_pays"] != "FR")]
        agg = (tx.groupby("code_pays")
                 .agg(montant=("montant", "sum"), nb_operations=("transaction_id", "count"),
                      nb_clients=("numero_dossier", "nunique"))
                 .reset_index())
    _, pays = charger_referentiels()
    agg = agg.merge(pays.rename(columns={"libelle": "pays", "niveau_risque": "risque_pays"}), on="code_pays", how="left")
    return agg.sort_values("montant", ascending=False).reset_index(drop=True)


def _signal(op) -> str:
    """Signal de risque visible par l'analyste, déduit des seuls référentiels."""
    signaux = []
    if op["risque_pays"] == "eleve":
        signaux.append("Pays à risque élevé")
    if op["famille"] in FAMILLES_RISQUE:
        signaux.append(FAMILLES_RISQUE[op["famille"]])
    if op["type_trans"] == "DEPOT_ESPECES" and 0.8 * SEUIL_ALERTE_ESPECES <= op["montant"] < SEUIL_ALERTE_ESPECES:
        signaux.append("Espèces sous le seuil")
    return " · ".join(signaux)
