"""Recherche dans la base de connaissances (RAG).

Deux moteurs, même interface :
- "vertex" : embeddings Vertex AI (modèle multilingue), similarité cosinus ;
  l'index est calculé une fois et sauvegardé dans data/rag/index.json ;
- "local"  : TF-IDF (scikit-learn), sans appel réseau : tests et mode hors ligne.

Le corpus est petit (une quinzaine de fiches), un index en mémoire suffit.
À plus grande échelle : BigQuery Vector Search ou Vertex AI Vector Search.
"""

import json
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.generate.config import PROJECT_ROOT

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "index.json"
MODELE_EMBEDDING = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")


@dataclass(frozen=True)
class Document:
    id: str
    titre: str
    source: str
    texte: str

    def pour_prompt(self) -> str:
        return f"[{self.id}] {self.titre}\nSource : {self.source}\n{self.texte}"


def charger_corpus() -> list:
    documents = []
    for chemin in sorted(CORPUS_DIR.glob("*.md")):
        if chemin.name == "README.md":
            continue
        entete, _, corps = chemin.read_text(encoding="utf-8").partition("\n\n")
        champs = dict(ligne.split(": ", 1) for ligne in entete.splitlines())
        documents.append(Document(champs["id"], champs["titre"], champs["source"], corps.strip()))
    return documents


# --------------------------------------------------------------------------
# Moteur local : TF-IDF
# --------------------------------------------------------------------------
class RechercheLocale:
    def __init__(self, documents):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents = documents
        self.vectoriseur = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
                                           sublinear_tf=True)
        self.matrice = self.vectoriseur.fit_transform([f"{d.titre} {d.texte}" for d in documents])

    def chercher(self, requete: str, k: int = 4) -> list:
        scores = (self.vectoriseur.transform([requete]) @ self.matrice.T).toarray()[0]
        ordre = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in ordre if scores[i] > 0]


# --------------------------------------------------------------------------
# Moteur Vertex AI : embeddings
# --------------------------------------------------------------------------
def _embeddings_vertex(textes: list, tache: str) -> np.ndarray:
    from langchain_google_vertexai import VertexAIEmbeddings

    modele = VertexAIEmbeddings(model_name=MODELE_EMBEDDING,
                                location=os.environ.get("EMBEDDING_LOCATION", "europe-west1"))
    if tache == "document":
        vecteurs = modele.embed_documents(textes)
    else:
        vecteurs = [modele.embed_query(t) for t in textes]
    matrice = np.array(vecteurs, dtype=float)
    return matrice / np.linalg.norm(matrice, axis=1, keepdims=True)


def construire_index_vertex() -> Path:
    documents = charger_corpus()
    vecteurs = _embeddings_vertex([f"{d.titre}\n{d.texte}" for d in documents], "document")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps({
        "modele": MODELE_EMBEDDING,
        "documents": [d.id for d in documents],
        "vecteurs": vecteurs.round(6).tolist(),
    }))
    return INDEX_PATH


class RechercheVertex:
    def __init__(self, documents):
        if not INDEX_PATH.exists():
            construire_index_vertex()
        index = json.loads(INDEX_PATH.read_text())
        par_id = {d.id: d for d in documents}
        if set(index["documents"]) != set(par_id):
            construire_index_vertex()  # le corpus a changé : on reconstruit
            index = json.loads(INDEX_PATH.read_text())
        self.documents = [par_id[i] for i in index["documents"]]
        self.vecteurs = np.array(index["vecteurs"])

    def chercher(self, requete: str, k: int = 4) -> list:
        q = _embeddings_vertex([requete], "requete")[0]
        scores = self.vecteurs @ q
        ordre = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in ordre]


@lru_cache(maxsize=2)
def moteur(nom: str = None):
    """Moteur Vertex par défaut ; repli automatique sur TF-IDF si Vertex est injoignable."""
    nom = nom or os.environ.get("RAG_MOTEUR", "vertex")
    documents = charger_corpus()
    if nom != "vertex":
        return RechercheLocale(documents)
    try:
        return RechercheVertex(documents)
    except Exception as erreur:  # noqa: BLE001
        warnings.warn(f"Embeddings Vertex indisponibles ({erreur}) : recherche locale TF-IDF utilisée.")
        return RechercheLocale(documents)


if __name__ == "__main__":
    print(f"Index construit : {construire_index_vertex()}")
