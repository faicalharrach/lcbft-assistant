"""Détection hybride : Isolation Forest (50 %) + règles métier (50 %), expliquée par SHAP.

Pourquoi un hybride :
- les règles encodent les typologies connues et sont lisibles par la conformité,
  mais elles ratent tout comportement qu'on n'a pas prévu ;
- l'Isolation Forest repère les profils atypiques sans étiquettes (cas réel en
  LCB-FT), mais seul il est difficile à justifier devant un auditeur.
Le score final combine les deux, et chaque alerte est accompagnée de ses raisons :
règles déclenchées + variables qui ont le plus pesé dans l'anomalie (SHAP).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from src.generate.config import SEED

POIDS_IF = 0.5
POIDS_REGLES = 0.5

# Variables vues par l'Isolation Forest. Exclues : identifiants, CSP (texte),
# nationalité (biais), et avg_montant_ccsp (constante par CSP).
FEATURES_MODELE = [
    "age", "revenu_mensuel", "anciennete_mois",
    "nb_trans", "montant_total_debit", "montant_total_credit",
    "nb_achats", "montant_achats", "nb_dab", "montant_dab",
    "nb_depots_especes", "montant_depots_especes", "montant_max_depot_especes",
    "nb_depots_proches_seuil", "montant_max_virement_entrant",
    "nb_trans_pays_risque_eleve", "montant_pays_risque_eleve",
    "nb_trans_pays_risque_moyen", "nb_pays_etrangers",
    "nb_trans_jeux", "montant_jeux", "nb_trans_crypto", "montant_crypto",
    "nb_trans_valeurs", "montant_valeurs", "nb_trans_transfert_prepaye",
    "nb_allers_retours_48h", "max_occu",
    "ratio_credits_revenu", "ratio_depenses_revenu", "part_especes_credits",
    "part_mcc_risque_debits", "ecart_montant_ccsp", "ratio_depense_ccsp",
]

# Libellés lisibles pour l'interface et la note d'analyse.
LIBELLES = {
    "nb_depots_proches_seuil": "Dépôts d'espèces juste sous le seuil",
    "nb_allers_retours_48h": "Entrées ressorties en moins de 48h",
    "nb_trans_pays_risque_eleve": "Opérations avec des pays à risque élevé",
    "montant_pays_risque_eleve": "Montant échangé avec des pays à risque élevé",
    "ratio_credits_revenu": "Entrées d'argent rapportées au revenu déclaré",
    "ratio_depenses_revenu": "Dépenses rapportées au revenu déclaré",
    "montant_crypto": "Montant d'achats de crypto-actifs",
    "nb_trans_crypto": "Nombre d'achats de crypto-actifs",
    "montant_valeurs": "Achats de biens de valeur (bijoux, art, métaux)",
    "part_especes_credits": "Part des espèces dans les entrées",
    "montant_depots_especes": "Montant des dépôts d'espèces",
    "montant_max_virement_entrant": "Plus gros virement reçu",
    "ratio_depense_ccsp": "Dépenses comparées à la moyenne de la CSP",
    "ecart_montant_ccsp": "Écart de dépenses à la moyenne de la CSP",
    "montant_total_credit": "Total des entrées",
    "montant_total_debit": "Total des sorties",
    "age": "Âge",
    "revenu_mensuel": "Revenu mensuel déclaré",
    "anciennete_mois": "Ancienneté du compte",
    "nb_trans": "Nombre total d'opérations",
    "nb_achats": "Nombre d'achats carte",
    "montant_achats": "Montant des achats carte",
    "nb_dab": "Nombre de retraits DAB",
    "montant_dab": "Montant des retraits DAB",
    "nb_depots_especes": "Nombre de dépôts d'espèces",
    "montant_max_depot_especes": "Plus gros dépôt d'espèces",
    "nb_trans_pays_risque_moyen": "Opérations avec des pays à risque moyen",
    "nb_pays_etrangers": "Nombre de pays étrangers",
    "nb_trans_jeux": "Nombre d'opérations de jeux d'argent",
    "montant_jeux": "Montant des jeux d'argent",
    "nb_trans_valeurs": "Nombre d'achats de biens de valeur",
    "nb_trans_transfert_prepaye": "Transferts d'argent et cartes prépayées",
    "max_occu": "Répétition d'un même montant",
    "part_mcc_risque_debits": "Part des dépenses chez des marchands à risque",
}


# --------------------------------------------------------------------------
# Règles métier
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Regle:
    code: str
    libelle: str
    points: int
    condition: callable


REGLES = [
    Regle("R1_FRACTIONNEMENT", "Au moins 4 dépôts d'espèces juste sous le seuil d'alerte", 40,
          lambda f: f["nb_depots_proches_seuil"] >= 4),
    Regle("R2_COMPTE_DE_PASSAGE", "Au moins 2 entrées importantes ressorties en moins de 48h", 40,
          lambda f: f["nb_allers_retours_48h"] >= 2),
    Regle("R3_PAYS_A_RISQUE", "Flux répétés ou importants avec des pays à risque élevé", 35,
          lambda f: (f["nb_trans_pays_risque_eleve"] >= 4) | (f["montant_pays_risque_eleve"] >= 5_000)),
    Regle("R4_INCOHERENCE_REVENUS", "Entrées d'argent au moins 2 fois supérieures au revenu déclaré", 30,
          lambda f: f["ratio_credits_revenu"] >= 2),
    Regle("R5_CRYPTO", "Achats de crypto-actifs élevés au regard du revenu", 35,
          lambda f: (f["montant_crypto"] >= 5_000)
          & (f["montant_crypto"] >= 0.3 * f["revenu_annuel_reference"])),
    Regle("R6_ESPECES", "Part importante d'espèces hors profession qui en manipule", 20,
          lambda f: (f["part_especes_credits"] >= 0.3) & ~f["ccsp"].str.startswith("Artisans")),
    Regle("R7_BIENS_DE_VALEUR", "Achats de biens de valeur élevés au regard du revenu", 20,
          lambda f: (f["montant_valeurs"] >= 3_000)
          & (f["montant_valeurs"] >= 0.3 * f["revenu_annuel_reference"])),
    Regle("R8_SUR_DEPENSE_CSP", "Dépenses plus de 2,5 fois supérieures à la moyenne de la CSP", 15,
          lambda f: f["ratio_depense_ccsp"] >= 2.5),
]


def appliquer_regles(features: pd.DataFrame) -> pd.DataFrame:
    """Une colonne booléenne par règle, plus le score règles (plafonné à 100)."""
    declenchees = pd.DataFrame({r.code: r.condition(features).astype(bool) for r in REGLES},
                               index=features.index)
    points = np.array([r.points for r in REGLES])
    declenchees["score_regles"] = np.minimum(declenchees.to_numpy() @ points, 100)
    return declenchees


# --------------------------------------------------------------------------
# Isolation Forest
# --------------------------------------------------------------------------
def preparer(features: pd.DataFrame) -> pd.DataFrame:
    """Log signé : les montants ont des queues très longues, le log les resserre
    pour que les coupures aléatoires de l'Isolation Forest soient mieux réparties."""
    X = features[FEATURES_MODELE].astype(float)
    return np.sign(X) * np.log1p(X.abs())


def entrainer_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    modele = IsolationForest(n_estimators=300, max_samples=256, contamination="auto",
                             random_state=SEED, n_jobs=-1)
    return modele.fit(X)


def score_percentile(valeurs: np.ndarray) -> np.ndarray:
    """Rang en percentile (0-100) : robuste aux valeurs extrêmes et comparable aux règles."""
    return pd.Series(valeurs).rank(pct=True).to_numpy() * 100


# --------------------------------------------------------------------------
# Pipeline complet
# --------------------------------------------------------------------------
def scorer(features: pd.DataFrame, top_raisons: int = 3) -> pd.DataFrame:
    features = features.reset_index(drop=True)
    X = preparer(features)
    modele = entrainer_isolation_forest(X)

    # decision_function : plus c'est bas, plus c'est anormal -> on inverse.
    score_if = score_percentile(-modele.decision_function(X))
    regles = appliquer_regles(features)

    # SHAP sur l'Isolation Forest : une valeur négative raccourcit le chemin
    # d'isolement, donc rend le client PLUS anormal. On inverse le signe pour
    # lire "contribution au risque".
    contributions = -shap.TreeExplainer(modele).shap_values(X)

    raisons_modele = []
    for ligne in contributions:
        ordre = np.argsort(ligne)[::-1][:top_raisons]
        raisons_modele.append([FEATURES_MODELE[j] for j in ordre if ligne[j] > 0])

    regles_declenchees = [
        [r.code for r in REGLES if regles.at[i, r.code]] for i in range(len(features))
    ]

    resultat = pd.DataFrame({
        "numero_dossier": features["numero_dossier"],
        "score_if": score_if.round(1),
        "score_regles": regles["score_regles"].astype(float),
        "score_final": (POIDS_IF * score_if + POIDS_REGLES * regles["score_regles"]).round(1),
        "regles_declenchees": regles_declenchees,
        "raisons_modele": raisons_modele,
    })
    resultat["rang"] = resultat["score_final"].rank(ascending=False, method="first").astype(int)
    shap_df = pd.DataFrame(contributions, columns=[f"shap_{c}" for c in FEATURES_MODELE])
    return resultat.sort_values("rang").reset_index(drop=True), shap_df.loc[resultat.sort_values("rang").index].reset_index(drop=True)
