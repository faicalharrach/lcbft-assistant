"""Tests de la détection hybride."""

import pandas as pd
import pytest

from src.data import charger_features
from src.detection.model import FEATURES_MODELE, REGLES, appliquer_regles, scorer
from src.generate.config import RAW_DIR


@pytest.fixture(scope="module")
def features():
    if not (RAW_DIR / "transactions.csv").exists():
        pytest.skip("Lancer d'abord python -m src.generate.run_all")
    return charger_features("local")


@pytest.fixture(scope="module")
def resultat(features):
    return scorer(features.sample(800, random_state=0))


def test_scores_entre_0_et_100(resultat):
    scores, _ = resultat
    for col in ["score_if", "score_regles", "score_final"]:
        assert scores[col].between(0, 100).all()


def test_score_final_est_la_moyenne(resultat):
    scores, _ = resultat
    attendu = (0.5 * scores["score_if"] + 0.5 * scores["score_regles"]).round(1)
    # Écart possible de 0,15 : score_if est arrondi après le calcul du score final.
    assert (scores["score_final"] - attendu).abs().max() <= 0.15


def test_rang_suit_le_score(resultat):
    scores, _ = resultat
    assert scores["rang"].is_monotonic_increasing
    assert scores["score_final"].is_monotonic_decreasing


def test_shap_aligne_sur_les_scores(resultat):
    scores, shap_df = resultat
    assert len(shap_df) == len(scores)
    assert list(shap_df.columns) == [f"shap_{c}" for c in FEATURES_MODELE]
    # Plus la somme des contributions est forte, plus le client est anormal.
    assert shap_df.sum(axis=1).corr(scores["score_if"]) > 0.8


def test_regle_fractionnement():
    base = {c: 0.0 for c in FEATURES_MODELE}
    base.update(revenu_annuel_reference=24_000.0, ccsp="Employés")
    normal = dict(base, nb_depots_proches_seuil=1)
    suspect = dict(base, nb_depots_proches_seuil=6)
    resultat = appliquer_regles(pd.DataFrame([normal, suspect]))
    assert list(resultat["R1_FRACTIONNEMENT"]) == [False, True]
    assert resultat.loc[1, "score_regles"] == next(r.points for r in REGLES if r.code == "R1_FRACTIONNEMENT")


def test_artisan_exempte_de_la_regle_especes():
    base = {c: 0.0 for c in FEATURES_MODELE}
    base.update(revenu_annuel_reference=30_000.0, part_especes_credits=0.5)
    resultat = appliquer_regles(pd.DataFrame([
        dict(base, ccsp="Artisans, commerçants, chefs d'entreprise"),
        dict(base, ccsp="Employés"),
    ]))
    assert list(resultat["R6_ESPECES"]) == [False, True]
