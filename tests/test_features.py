"""Tests de la requête de features, exécutée en local sur DuckDB.

Prérequis : python -m src.generate.run_all (les CSV de data/raw/ doivent exister).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.generate.config import PROJECT_ROOT, RAW_DIR
from src.generate.transactions import SEUIL_ALERTE_ESPECES
from tests.sql_local import executer

SQL = (PROJECT_ROOT / "sql" / "features_clients.sql").read_text()


@pytest.fixture(scope="module")
def features():
    if not (RAW_DIR / "transactions.csv").exists():
        pytest.skip("Lancer d'abord python -m src.generate.run_all")
    df = executer(SQL.format(raw="raw", seuil_especes=SEUIL_ALERTE_ESPECES))
    labels = pd.read_csv(RAW_DIR / "labels_clients.csv")[["numero_dossier", "typologie"]]
    return df.merge(labels, on="numero_dossier", how="left").fillna({"typologie": "normal"})


def test_une_ligne_par_client(features):
    clients = pd.read_csv(RAW_DIR / "clients.csv")
    assert len(features) == len(clients)
    assert features["numero_dossier"].is_unique


def test_pas_de_valeur_manquante(features):
    manquantes = features.isna().sum()
    assert manquantes.sum() == 0, manquantes[manquantes > 0].to_dict()


def test_pas_de_nationalite_dans_les_features(features):
    assert "nationalite" not in features.columns
    assert "pays_residence" not in features.columns


def _mediane(features, typologie, colonne):
    return features.loc[features["typologie"] == typologie, colonne].median()


@pytest.mark.parametrize("typologie, colonne", [
    ("fractionnement", "nb_depots_proches_seuil"),
    ("compte_de_passage", "nb_allers_retours_48h"),
    ("pays_a_risque", "nb_trans_pays_risque_eleve"),
    ("incoherence_revenus", "ratio_credits_revenu"),
    ("crypto", "montant_crypto"),
])
def test_chaque_typologie_a_un_signal(features, typologie, colonne):
    """La feature dédiée à une typologie est plus élevée chez ses porteurs."""
    assert _mediane(features, typologie, colonne) > _mediane(features, "normal", colonne)
