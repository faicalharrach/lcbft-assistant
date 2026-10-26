"""Vérifie en une commande que Vertex AI répond (Gemini et embeddings).

    python -m src.agent.diagnostic

Modèles et régions se changent par variables d'environnement :
GEMINI_MODEL, VERTEX_LOCATION, EMBEDDING_MODEL, EMBEDDING_LOCATION.
"""

import os

from src.agent.graphe import MODELE_PAR_DEFAUT
from src.agent.rag import MODELE_EMBEDDING

if __name__ == "__main__":
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

    modele = os.environ.get("GEMINI_MODEL", MODELE_PAR_DEFAUT)
    region = os.environ.get("VERTEX_LOCATION", "global")
    try:
        reponse = ChatVertexAI(model=modele, location=region, max_output_tokens=20).invoke(
            "Réponds seulement : OK")
        print(f"[OK] Gemini {modele} ({region}) : {reponse.content!r}")
    except Exception as erreur:  # noqa: BLE001
        print(f"[ÉCHEC] Gemini {modele} ({region}) : {erreur}")
        print("        Essayez GEMINI_MODEL=gemini-3.8-flash ou VERTEX_LOCATION=europe-west1.")

    region_emb = os.environ.get("EMBEDDING_LOCATION", "europe-west1")
    try:
        vecteur = VertexAIEmbeddings(model_name=MODELE_EMBEDDING, location=region_emb).embed_query("test")
        print(f"[OK] Embeddings {MODELE_EMBEDDING} ({region_emb}) : vecteur de dimension {len(vecteur)}")
    except Exception as erreur:  # noqa: BLE001
        print(f"[ÉCHEC] Embeddings {MODELE_EMBEDDING} ({region_emb}) : {erreur}")
        print("        L'agent utilisera la recherche locale TF-IDF à la place.")
