"""Table clients : profils synthétiques cohérents (âge, CSP, revenus, foyer).

Choix de modélisation :
- les revenus sont MENSUELS et nets, tirés selon une loi log-normale
  dont la médiane dépend de la CSP (un cadre gagne plus qu'un employé) ;
- l'âge dépend aussi de la CSP (un étudiant a 18-26 ans, un retraité 62+) ;
- environ 30 % des clients actifs ou retraités vivent en couple et partagent
  un numéro de foyer, d'où revenu_foyer >= revenus propres ;
- la nationalité est conservée pour le réalisme de la fiche client, mais elle
  ne sera PAS utilisée comme variable du modèle (risque de biais
  discriminatoire). Le risque pays passe par les flux vers des pays à risque.
"""

import numpy as np
import pandas as pd
from faker import Faker

from src.generate.config import N_CLIENTS, SEED, get_rng, save

DATE_REFERENCE = pd.Timestamp("2025-01-01")
OUVERTURE_MIN = pd.Timestamp("2005-01-01")
OUVERTURE_MAX = pd.Timestamp("2024-12-31")

# csp : (part de la clientèle, age_min, age_max, salaire médian, sigma, pension médiane)
CSP = {
    "Cadres": (0.15, 24, 64, 4000, 0.35, 0),
    "Professions intermédiaires": (0.17, 22, 64, 2400, 0.25, 0),
    "Employés": (0.17, 20, 64, 1650, 0.20, 0),
    "Ouvriers": (0.12, 20, 64, 1700, 0.20, 0),
    "Artisans, commerçants, chefs d'entreprise": (0.06, 25, 66, 2600, 0.60, 0),
    "Agriculteurs": (0.01, 25, 66, 1600, 0.50, 0),
    "Retraités": (0.22, 62, 92, 0, 0.0, 1500),
    "Étudiants": (0.06, 18, 26, 450, 0.60, 0),
    "Sans activité": (0.04, 18, 64, 0, 0.0, 0),
}
SIGMA_PENSION = 0.35
PART_ETUDIANTS_SANS_JOB = 0.5
PART_EN_COUPLE = 0.30

NATIONALITES = {
    "FR": 0.86, "MA": 0.03, "DZ": 0.02, "PT": 0.02, "TN": 0.015, "IT": 0.01,
    "ES": 0.01, "SN": 0.01, "TR": 0.01, "CN": 0.01, "BE": 0.005,
}
RESIDENCES = {"FR": 0.97, "BE": 0.01, "CH": 0.01, "ES": 0.005, "MA": 0.005}


def _tirer(rng, distribution: dict, n: int) -> np.ndarray:
    valeurs = list(distribution)
    probas = np.array(list(distribution.values()), dtype=float)
    return rng.choice(valeurs, size=n, p=probas / probas.sum())


def _lognormal(rng, mediane: float, sigma: float, n: int) -> np.ndarray:
    if mediane <= 0 or n == 0:
        return np.zeros(n)
    # La médiane d'une log-normale vaut exp(mu), donc mu = log(médiane).
    return np.round(rng.lognormal(np.log(mediane), sigma, n), -1)


def _attribuer_foyers(rng, df: pd.DataFrame) -> pd.Series:
    """Un foyer par client, puis on regroupe des paires de clients en couples."""
    foyer = pd.Series(np.arange(len(df)), index=df.index)
    eligibles = df.index[df["ccsp"] != "Étudiants"].to_numpy()
    n_couples = int(len(eligibles) * PART_EN_COUPLE / 2)
    choisis = rng.permutation(eligibles)[: 2 * n_couples].reshape(-1, 2)
    for a, b in choisis:
        foyer[b] = foyer[a]
    # Renumérotation propre : F000001, F000002...
    codes = pd.factorize(foyer)[0] + 1
    return pd.Series([f"F{c:06d}" for c in codes], index=df.index)


def _dates_ouverture(rng, ages: np.ndarray) -> pd.Series:
    """Ouverture entre 2005 et fin 2024, jamais avant les 18 ans du client."""
    dates = []
    for age in ages:
        majorite = DATE_REFERENCE - pd.DateOffset(years=int(age) - 18)
        debut = max(OUVERTURE_MIN, majorite)
        jours = max((OUVERTURE_MAX - debut).days, 0)
        dates.append(debut + pd.Timedelta(days=int(rng.integers(0, jours + 1))))
    return pd.Series(dates).dt.date


def build_clients(n: int = N_CLIENTS) -> pd.DataFrame:
    rng = get_rng(offset=2)
    fake = Faker("fr_FR")
    fake.seed_instance(SEED + 2)

    parts = {csp: params[0] for csp, params in CSP.items()}
    df = pd.DataFrame({"ccsp": _tirer(rng, parts, n)})

    df["age"] = 0
    df["salaire_declare"] = 0.0
    df["pension"] = 0.0
    for csp, (_, age_min, age_max, sal_med, sigma, pens_med) in CSP.items():
        idx = df.index[df["ccsp"] == csp]
        k = len(idx)
        df.loc[idx, "age"] = rng.integers(age_min, age_max + 1, k)
        df.loc[idx, "salaire_declare"] = _lognormal(rng, sal_med, sigma, k)
        df.loc[idx, "pension"] = _lognormal(rng, pens_med, SIGMA_PENSION, k)

    # La moitié des étudiants n'a pas de job déclaré.
    etudiants = df.index[df["ccsp"] == "Étudiants"]
    sans_job = rng.random(len(etudiants)) < PART_ETUDIANTS_SANS_JOB
    df.loc[etudiants[sans_job], "salaire_declare"] = 0.0

    df["numero_dossier"] = [f"D{i + 1:06d}" for i in range(n)]
    df["numero_foyer"] = _attribuer_foyers(rng, df)
    revenus = df["salaire_declare"] + df["pension"]
    df["revenu_foyer"] = revenus.groupby(df["numero_foyer"]).transform("sum")

    df["nom"] = [fake.last_name() for _ in range(n)]
    df["prenom"] = [fake.first_name() for _ in range(n)]
    df["nationalite"] = _tirer(rng, NATIONALITES, n)
    df["pays_residence"] = _tirer(rng, RESIDENCES, n)
    df["date_ouverture"] = _dates_ouverture(rng, df["age"].to_numpy())

    colonnes = [
        "numero_dossier", "numero_foyer", "nom", "prenom", "age", "nationalite",
        "ccsp", "salaire_declare", "pension", "revenu_foyer", "pays_residence",
        "date_ouverture",
    ]
    df = df[colonnes]
    assert df["numero_dossier"].is_unique
    assert (df["revenu_foyer"] >= df["salaire_declare"] + df["pension"]).all()
    return df


if __name__ == "__main__":
    df = build_clients()
    path = save(df, "clients")
    resume = df.groupby("ccsp").agg(
        nb=("numero_dossier", "count"),
        age_moyen=("age", "mean"),
        salaire_median=("salaire_declare", "median"),
        pension_mediane=("pension", "median"),
    ).round(0).sort_values("salaire_median", ascending=False)
    print(resume.to_string())
    print(f"\nFoyers : {df['numero_foyer'].nunique()} pour {len(df)} clients")
    print(f"{len(df)} clients -> {path}")
