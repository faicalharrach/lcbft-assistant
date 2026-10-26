"""Référentiel pays avec un niveau de risque LCB-FT.

Liste ILLUSTRATIVE, inspirée des publications du GAFI :
- "eleve"  : juridictions sous appel à l'action (liste noire) ou sous
             surveillance renforcée (liste grise) ;
- "moyen"  : places financières ou centres offshore surveillés par la conformité ;
- "faible" : le reste.
Les listes du GAFI changent plusieurs fois par an : vérifier la dernière
publication avant toute utilisation hors démonstration.
"""

import pandas as pd

from src.generate.config import save

COLUMNS = ["code_pays", "libelle", "niveau_risque"]

PAYS = [
    # --- Risque faible ---
    ("FR", "France", "faible"),
    ("BE", "Belgique", "faible"),
    ("DE", "Allemagne", "faible"),
    ("ES", "Espagne", "faible"),
    ("IT", "Italie", "faible"),
    ("PT", "Portugal", "faible"),
    ("NL", "Pays-Bas", "faible"),
    ("CH", "Suisse", "faible"),
    ("GB", "Royaume-Uni", "faible"),
    ("US", "États-Unis", "faible"),
    ("CA", "Canada", "faible"),
    ("MA", "Maroc", "faible"),
    ("TN", "Tunisie", "faible"),
    ("SN", "Sénégal", "faible"),
    ("TR", "Turquie", "faible"),
    ("CN", "Chine", "faible"),
    # --- Risque moyen ---
    ("MC", "Monaco", "moyen"),
    ("AE", "Émirats arabes unis", "moyen"),
    ("PA", "Panama", "moyen"),
    ("KY", "Îles Caïmans", "moyen"),
    ("CI", "Côte d'Ivoire", "moyen"),
    # --- Risque élevé ---
    ("DZ", "Algérie", "eleve"),
    ("LB", "Liban", "eleve"),
    ("VE", "Venezuela", "eleve"),
    ("YE", "Yémen", "eleve"),
    ("SY", "Syrie", "eleve"),
    ("HT", "Haïti", "eleve"),
    ("KP", "Corée du Nord", "eleve"),
    ("IR", "Iran", "eleve"),
    ("MM", "Myanmar", "eleve"),
]


def build_ref_pays() -> pd.DataFrame:
    df = pd.DataFrame(PAYS, columns=COLUMNS)
    assert df["code_pays"].is_unique, "Doublon de code pays dans le référentiel"
    assert set(df["niveau_risque"]) <= {"faible", "moyen", "eleve"}
    return df


if __name__ == "__main__":
    df = build_ref_pays()
    path = save(df, "ref_pays")
    print(df["niveau_risque"].value_counts().to_string())
    print(f"\n{len(df)} pays -> {path}")
