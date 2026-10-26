"""Évaluation de la détection contre la vérité terrain.

On compare trois classements : Isolation Forest seul, règles seules, hybride.
Métriques orientées "file d'alertes" : l'analyste traite les K premiers dossiers.
- précision@K : part de vrais suspects parmi les K premiers ;
- rappel@K    : part des suspects retrouvés dans les K premiers ;
- rappel par typologie et intensité : où chaque méthode est forte ou aveugle ;
- faux amis dans la file : quels clients légitimes coûtent du temps aux analystes.

Limite assumée : les données et les règles ont été écrites par la même personne.
Les cas "discrets" et les "faux amis" réduisent ce biais sans le supprimer.
"""

import pandas as pd

from src.generate.config import PROJECT_ROOT

SCORES = {"score_if": "Isolation Forest", "score_regles": "Règles", "score_final": "Hybride"}
K_LISTE = [100, 200, 300]
K_DETAIL = 300


def preparer_evaluation(scores, labels, profils=None) -> pd.DataFrame:
    df = scores.merge(labels, on="numero_dossier", how="left")
    if profils is not None:
        df = df.merge(profils, on="numero_dossier", how="left")
    df["suspect"] = df["typologie"].notna()
    return df


def metriques_globales(df: pd.DataFrame) -> pd.DataFrame:
    n_suspects = df["suspect"].sum()
    lignes = []
    for col, nom in SCORES.items():
        ordre = df.sort_values(col, ascending=False)
        ligne = {"methode": nom}
        for k in K_LISTE:
            top = ordre.head(k)
            ligne[f"precision@{k}"] = round(top["suspect"].mean(), 2)
            ligne[f"rappel@{k}"] = round(top["suspect"].sum() / n_suspects, 2)
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("methode")


def rappel_par_typologie(df: pd.DataFrame, k: int = K_DETAIL) -> pd.DataFrame:
    suspects = df[df["suspect"]]
    colonnes = {}
    for col, nom in SCORES.items():
        top = set(df.sort_values(col, ascending=False).head(k)["numero_dossier"])
        trouve = suspects["numero_dossier"].isin(top)
        colonnes[nom] = trouve.groupby([suspects["typologie"], suspects["intensite"]]).mean().round(2)
    return pd.DataFrame(colonnes)


def faux_amis_dans_la_file(df: pd.DataFrame, k: int = K_DETAIL) -> pd.Series:
    if "profil" not in df.columns:
        return pd.Series(dtype=int)
    top = df.sort_values("score_final", ascending=False).head(k)
    return top["profil"].value_counts()


def rapport(df: pd.DataFrame) -> str:
    globales = metriques_globales(df)
    detail = rappel_par_typologie(df)
    faux_amis = faux_amis_dans_la_file(df)
    lignes = [
        "# Évaluation de la détection",
        "",
        f"{len(df):,} clients, dont {int(df['suspect'].sum())} suspects (vérité terrain).",
        "",
        "## Métriques globales",
        "",
        globales.to_markdown(),
        "",
        f"## Rappel par typologie et intensité (top {K_DETAIL})",
        "",
        detail.to_markdown(),
        "",
        f"## Clients légitimes atypiques dans le top {K_DETAIL} (hybride)",
        "",
        faux_amis.to_frame("nb_clients").to_markdown() if len(faux_amis) else "Aucun.",
        "",
    ]
    return "\n".join(lignes)


def sauver_rapport(texte: str):
    chemin = PROJECT_ROOT / "reports" / "evaluation.md"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(texte)
    return chemin
