"""Tests de cohérence des données générées. Lancer avec : pytest -q"""

import pandas as pd
import pytest

from src.generate.clients import build_clients
from src.generate.ref_marchands import build_ref_marchands
from src.generate.ref_mcc import build_ref_mcc
from src.generate.ref_pays import build_ref_pays


@pytest.fixture(scope="module")
def clients():
    return build_clients(n=2_000)


def test_referentiels_sans_doublon():
    assert build_ref_mcc()["mcc"].is_unique
    assert build_ref_pays()["code_pays"].is_unique


def test_marchands_majoritairement_courants():
    marchands = build_ref_marchands().merge(build_ref_mcc(), on="mcc")
    part_risque = (marchands["niveau_risque"] != "faible").mean()
    assert 0.05 <= part_risque <= 0.15


def test_generation_reproductible():
    pd.testing.assert_frame_equal(build_clients(n=200), build_clients(n=200))


def test_salaire_coherent_avec_csp(clients):
    medianes = clients.groupby("ccsp")["salaire_declare"].median()
    assert medianes["Cadres"] > medianes["Professions intermédiaires"] > medianes["Employés"]


def test_ages_coherents_avec_csp(clients):
    assert clients.loc[clients["ccsp"] == "Étudiants", "age"].max() <= 26
    assert clients.loc[clients["ccsp"] == "Retraités", "age"].min() >= 62


def test_seuls_les_retraites_ont_une_pension(clients):
    non_retraites = clients["ccsp"] != "Retraités"
    assert (clients.loc[non_retraites, "pension"] == 0).all()
    assert (clients.loc[~non_retraites, "pension"] > 0).all()


def test_ouverture_apres_majorite(clients):
    ouverture = pd.to_datetime(clients["date_ouverture"])
    naissance_max = pd.Timestamp("2025-01-01") - pd.to_timedelta(clients["age"] * 365.25, unit="D")
    assert (ouverture >= naissance_max + pd.DateOffset(years=17)).all()


def test_revenu_foyer(clients):
    revenus = clients["salaire_declare"] + clients["pension"]
    recalcule = revenus.groupby(clients["numero_foyer"]).transform("sum")
    assert (recalcule == clients["revenu_foyer"]).all()
    assert clients.groupby("numero_foyer").size().max() <= 2


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------
from src.generate.transactions import SEUIL_ALERTE_ESPECES, build_transactions  # noqa: E402


@pytest.fixture(scope="module")
def tx():
    clients = build_clients(n=2_000)
    transactions, labels_clients, labels_tx, _ = build_transactions(
        clients, build_ref_marchands(), build_ref_mcc(), build_ref_pays()
    )
    merged = transactions.merge(labels_tx[["transaction_id", "typologie"]], on="transaction_id", how="left")
    return transactions, labels_clients, merged


def test_transactions_sur_2025(tx):
    transactions, _, _ = tx
    annees = pd.to_datetime(transactions["date_trans"]).dt.year
    assert (annees == 2025).all()


def test_seuls_les_achats_ont_un_marchand(tx):
    transactions, _, _ = tx
    est_achat = transactions["type_trans"] == "ACHAT"
    assert transactions.loc[est_achat, "marchand_id"].notna().all()
    assert transactions.loc[~est_achat, "marchand_id"].isna().all()


def test_cinq_typologies_equilibrees(tx):
    _, labels_clients, _ = tx
    assert labels_clients["typologie"].value_counts().nunique() == 1
    assert labels_clients["typologie"].nunique() == 5


def test_fractionnement_sous_le_seuil(tx):
    _, _, merged = tx
    depots = merged[(merged["typologie"] == "fractionnement") & (merged["type_trans"] == "DEPOT_ESPECES")]
    assert (depots["montant"] < SEUIL_ALERTE_ESPECES).all()


def test_crypto_aussi_chez_des_clients_normaux(tx):
    """Sinon le modèle apprendrait que crypto = suspect."""
    _, labels_clients, merged = tx
    crypto = merged[merged["marchand_id"].isin(
        build_ref_marchands().query("mcc == 6051")["marchand_id"]
    )]
    normaux = crypto[~crypto["numero_dossier"].isin(labels_clients["numero_dossier"])]
    assert normaux["numero_dossier"].nunique() > 50
