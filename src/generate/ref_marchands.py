"""Référentiel des marchands : chaque marchand a un code MCC et un pays.

Environ 90 % des marchands ont un MCC courant. Les marchands à risque
sont plus souvent situés à l'étranger, comme dans la réalité.
"""

import pandas as pd
from faker import Faker

from src.generate.config import N_MARCHANDS, SEED, get_rng, save
from src.generate.ref_mcc import build_ref_mcc
from src.generate.ref_pays import build_ref_pays

PART_MCC_RISQUE = 0.10      # part des marchands avec un MCC moyen ou élevé
PART_ETRANGER_COURANT = 0.05  # marchands courants situés hors de France
PART_ETRANGER_RISQUE = 0.40   # marchands à risque situés hors de France

# Noms fictifs pour les plateformes en ligne, plus lisibles dans l'interface.
NOMS_PLATEFORMES = {
    "crypto_change": ["CryptoX", "CoinFlow", "BitHarbor", "ChainSwap", "Satoshi Desk"],
    "jeux": ["LuckyBet", "RoyalSpin", "BetNova", "Jackpot Club", "TurfPlus"],
    "transfert": ["SendFast", "CashLink", "GlobeTransfer"],
    "prepaye": ["PrepaidGo", "CardTop"],
}


def _choisir_pays(rng, est_risque: bool, pays: pd.DataFrame) -> str:
    part_etranger = PART_ETRANGER_RISQUE if est_risque else PART_ETRANGER_COURANT
    if rng.random() >= part_etranger:
        return "FR"
    etrangers = pays.loc[pays["code_pays"] != "FR", "code_pays"].to_numpy()
    return str(rng.choice(etrangers))


def build_ref_marchands(n: int = N_MARCHANDS) -> pd.DataFrame:
    rng = get_rng(offset=1)
    fake = Faker("fr_FR")
    fake.seed_instance(SEED + 1)

    mcc = build_ref_mcc()
    pays = build_ref_pays()

    mcc_courants = mcc[mcc["niveau_risque"] == "faible"]
    mcc_risque = mcc[mcc["niveau_risque"] != "faible"]

    # Chaque MCC (courant ou à risque) est présent au moins une fois :
    # sinon un tirage malchanceux peut faire disparaître toute une famille
    # (aucune plateforme crypto, par exemple).
    n_risque = max(round(n * PART_MCC_RISQUE), len(mcc_risque))
    tirage = pd.concat(
        [
            mcc_courants,
            mcc_courants.sample(n - n_risque - len(mcc_courants), replace=True, random_state=rng),
            mcc_risque,
            mcc_risque.sample(n_risque - len(mcc_risque), replace=True, random_state=rng),
        ],
        ignore_index=True,
    )

    compteurs = {famille: 0 for famille in NOMS_PLATEFORMES}
    lignes = []
    for i, row in tirage.iterrows():
        est_risque = row["niveau_risque"] != "faible"
        famille = row["famille"]

        if famille in NOMS_PLATEFORMES:
            noms = NOMS_PLATEFORMES[famille]
            nom = noms[compteurs[famille] % len(noms)]
            compteurs[famille] += 1
        else:
            nom = fake.company()

        lignes.append(
            {
                "marchand_id": f"M{i + 1:04d}",
                "nom": nom,
                "mcc": int(row["mcc"]),
                "code_pays": _choisir_pays(rng, est_risque, pays),
            }
        )

    df = pd.DataFrame(lignes)
    assert df["marchand_id"].is_unique
    assert df["mcc"].isin(mcc["mcc"]).all(), "MCC inconnu du référentiel"
    assert df["code_pays"].isin(pays["code_pays"]).all(), "Pays inconnu du référentiel"
    return df


if __name__ == "__main__":
    df = build_ref_marchands()
    path = save(df, "ref_marchands")
    mcc = build_ref_mcc()
    resume = df.merge(mcc, on="mcc").groupby("niveau_risque").agg(
        nb_marchands=("marchand_id", "count"),
        part_hors_france=("code_pays", lambda s: round((s != "FR").mean(), 2)),
    )
    print(resume.to_string())
    print(f"\n{len(df)} marchands -> {path}")
