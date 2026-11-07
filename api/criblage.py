"""Criblage approché des noms contre le registre des gels (chargé depuis BigQuery)."""

import os
import re
import unicodedata

from rapidfuzz import fuzz


def normaliser(nom: str) -> str:
    """'Grigory  Shilkín-Jr.' -> 'GRIGORY SHILKIN JR' : sans accents, ponctuation ni casse."""
    sans_accents = "".join(c for c in unicodedata.normalize("NFKD", nom) if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", sans_accents.upper()).strip()


def score_noms(mots_requete: list[str], mots_candidat: list[str]) -> float:
    """Pour chaque mot cherché, sa meilleure ressemblance avec un mot du candidat ; moyenne sur 0-100.

    - l'ordre des mots ne compte pas  ('SHILKIN GRIGORY' = 'GRIGORY SHILKIN') ;
    - les mots en plus chez le candidat ne pénalisent pas (patronyme 'VLADIMIROVICH') ;
    - une faute ou une autre translittération coûte peu ('GRIGORI' ~ 'GRIGORY' = 86) ;
    - un mot sans équivalent fait chuter le score ('GRIGORY PETROV' -> 70).
    """
    if not mots_candidat:
        return 0.0
    return sum(max(fuzz.ratio(m, c) for c in mots_candidat) for m in mots_requete) / len(mots_requete)


class ListeGels:
    """Les noms à cribler et la fiche de chaque personne, gardés en mémoire."""

    def __init__(self, noms: list[dict], fiches: dict[int, dict], date_publication: str | None):
        self.noms = noms                       # [{id_registre, type_nom, nom_complet}]
        self.fiches = fiches                   # {id_registre: {nature, nom, nationalites, ...}}
        self.date_publication = date_publication
        self.index = [normaliser(n["nom_complet"]).split() for n in noms]  # noms découpés en mots

    @classmethod
    def depuis_bigquery(cls) -> "ListeGels":
        from google.cloud import bigquery

        client = bigquery.Client(project=os.environ.get("GCP_PROJECT"))
        noms = [dict(r) for r in client.query(
            "SELECT id_registre, type_nom, nom_complet FROM lcbft_gels.noms").result()]
        fiches, date_pub = {}, None
        for r in client.query(
                """SELECT id_registre, nature, nom, nationalites, dates_naissance,
                          fondements, CAST(date_publication AS STRING) AS date_publication
                   FROM lcbft_gels.registre""").result():
            fiches[r["id_registre"]] = {k: r[k] for k in ("nature", "nom", "nationalites",
                                                           "dates_naissance", "fondements")}
            date_pub = r["date_publication"]
        return cls(noms, fiches, date_pub)

    def rechercher(self, nom: str, seuil: int = 85, limite: int = 10) -> list[dict]:
        """Personnes dont un nom ressemble à `nom` (score 0-100), meilleure correspondance d'abord."""
        mots = normaliser(nom).split()
        if not mots:
            return []
        trouves = [(score, position) for position, candidat in enumerate(self.index)
                   if (score := score_noms(mots, candidat)) >= seuil]
        meilleurs: dict[int, dict] = {}
        for score, position in trouves:
            n = self.noms[position]
            ident = n["id_registre"]
            if ident not in meilleurs or score > meilleurs[ident]["score"]:
                meilleurs[ident] = {"id_registre": ident, "score": round(score, 1),
                                    "nom_trouve": n["nom_complet"], "type_nom": n["type_nom"],
                                    **self.fiches.get(ident, {})}
        return sorted(meilleurs.values(), key=lambda r: -r["score"])[:limite]