"""Génère la note d'analyse d'un dossier en ligne de commande.

Usage :
    python -m src.agent.run D001407             # Gemini sur Vertex AI
    AGENT_LLM=simule RAG_MOTEUR=local python -m src.agent.run D001407   # hors ligne
    python -m src.agent.rag                     # (re)construit l'index d'embeddings
"""

import sys

import pandas as pd

from src.agent.contexte import enrichir_transactions
from src.agent.graphe import generer_note
from src.generate.config import PROJECT_ROOT, RAW_DIR


def charger_dossier(dossier: str) -> tuple:
    scores = pd.read_csv(PROJECT_ROOT / "data" / "scores" / "scores_clients.csv")
    clients = pd.read_csv(RAW_DIR / "clients.csv")
    ligne = scores.merge(clients, on="numero_dossier").set_index("numero_dossier").loc[dossier]
    client = ligne.to_dict() | {"numero_dossier": dossier}
    for cle in ["regles_declenchees", "raisons_modele"]:
        valeur = client.get(cle)
        client[cle] = valeur.split("|") if isinstance(valeur, str) and valeur else []
    tx = pd.read_csv(RAW_DIR / "transactions.csv")
    return client, enrichir_transactions(tx[tx["numero_dossier"] == dossier])


if __name__ == "__main__":
    dossier = sys.argv[1] if len(sys.argv) > 1 else "D001407"
    client, tx = charger_dossier(dossier)
    resultat = generer_note(client, tx)
    print(resultat["note"])
    print("\n---")
    print(f"Essais de rédaction : {resultat['tentatives']}")
    print("Contrôles : " + ("OK" if not resultat["problemes"] else "; ".join(resultat["problemes"])))
