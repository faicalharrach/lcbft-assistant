"""Référentiel des codes MCC (Merchant Category Code, norme ISO 18245).

Le code MCC est public : il accompagne chaque paiement carte.
Le niveau de risque LCB-FT associé ici est une pondération propre au projet.
"""

import pandas as pd

from src.generate.config import save

COLUMNS = ["mcc", "libelle", "famille", "niveau_risque"]

MCC = [
    # --- Dépenses courantes ---
    (5411, "Supermarchés, épiceries", "courant", "faible"),
    (5812, "Restaurants", "courant", "faible"),
    (5814, "Restauration rapide", "courant", "faible"),
    (5541, "Stations-service", "courant", "faible"),
    (5912, "Pharmacies", "courant", "faible"),
    (5651, "Vêtements", "courant", "faible"),
    (5732, "Électronique", "courant", "faible"),
    (4111, "Transports en commun", "courant", "faible"),
    (4900, "Énergie, eau", "courant", "faible"),
    # --- Jeux d'argent ---
    (7995, "Paris, casinos, jeux d'argent", "jeux", "eleve"),
    (7801, "Jeux d'argent en ligne", "jeux", "eleve"),
    (7802, "Courses hippiques et canines", "jeux", "eleve"),
    (7800, "Loteries d'État", "jeux", "eleve"),
    # --- Conversion de fonds ---
    (6051, "Quasi-cash : crypto-actifs, devises, mandats", "crypto_change", "eleve"),
    (4829, "Transferts d'argent", "transfert", "eleve"),
    (6540, "Rechargement de cartes prépayées", "prepaye", "eleve"),
    # --- Biens de valeur ---
    (5944, "Bijouteries", "valeurs", "moyen"),
    (5094, "Pierres et métaux précieux", "valeurs", "moyen"),
    (5971, "Galeries et marchands d'art", "valeurs", "moyen"),
    (5933, "Prêteurs sur gages", "gages", "moyen"),
]


def build_ref_mcc() -> pd.DataFrame:
    df = pd.DataFrame(MCC, columns=COLUMNS)
    assert df["mcc"].is_unique, "Doublon de code MCC dans le référentiel"
    assert set(df["niveau_risque"]) <= {"faible", "moyen", "eleve"}
    return df


if __name__ == "__main__":
    df = build_ref_mcc()
    path = save(df, "ref_mcc")
    print(df.to_string(index=False))
    print(f"\n{len(df)} MCC -> {path}")
