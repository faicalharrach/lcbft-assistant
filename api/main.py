"""API LCB-FT : criblage des noms contre le registre national des gels.

Lancement local :
    uvicorn api.main:app --reload --port 8000
Documentation interactive : http://localhost:8000/docs
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from pydantic import BaseModel

from api.criblage import ListeGels

# Âge maximal de la copie en mémoire avant de relire BigQuery (fraîcheur des données)
DUREE_MAX_SECONDES = int(os.environ.get("RECHARGEMENT_SECONDES", 3600))

etat: dict = {}
verrou = threading.Lock()


def _charger() -> None:
    etat["liste"] = ListeGels.depuis_bigquery()
    etat["charge_a"] = time.monotonic()
    etat["charge_le"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


def liste_a_jour() -> ListeGels:
    """La liste en mémoire, relue depuis BigQuery si elle a plus de DUREE_MAX_SECONDES."""
    if time.monotonic() - etat["charge_a"] > DUREE_MAX_SECONDES:
        with verrou:  # une seule requête recharge, les autres attendent puis profitent du résultat
            if time.monotonic() - etat["charge_a"] > DUREE_MAX_SECONDES:
                try:
                    _charger()
                except Exception as erreur:  # BigQuery indisponible : on garde l'ancienne copie
                    print(f"Rechargement échoué, ancienne liste conservée : {erreur}")
                    etat["charge_a"] = time.monotonic()
    return etat["liste"]


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    """Au démarrage : charge le registre une première fois en mémoire."""
    _charger()
    yield
    etat.clear()


app = FastAPI(
    title="API LCB-FT",
    description="Criblage des noms contre le registre national des gels des avoirs (DG Trésor).",
    version="0.2.0",
    lifespan=cycle_de_vie,
)


class Correspondance(BaseModel):
    id_registre: int
    score: float
    nom_trouve: str
    type_nom: str
    nature: str | None = None
    nom: str | None = None
    nationalites: list[str] = []
    dates_naissance: list[str] = []
    fondements: list[str] = []


class ResultatCriblage(BaseModel):
    nom_recherche: str
    seuil: int
    date_registre: str | None
    nb_correspondances: int
    alerte: bool
    correspondances: list[Correspondance]


@app.get("/sante")
def sante() -> dict:
    """Vérifie que l'API tourne, que le registre est chargé et depuis quand."""
    liste = liste_a_jour()
    return {
        "statut": "ok",
        "noms_charges": len(liste.noms),
        "date_registre": liste.date_publication,
        "copie_chargee_le": etat["charge_le"],
        "age_copie_minutes": round((time.monotonic() - etat["charge_a"]) / 60, 1),
    }


@app.get("/criblage", response_model=ResultatCriblage)
def criblage(
    nom: str = Query(..., min_length=2, description="Nom complet à cribler, ex. 'Grigory Shilkin'"),
    seuil: int = Query(85, ge=50, le=100, description="Score minimal de ressemblance (0-100)"),
) -> ResultatCriblage:
    """Cherche les personnes du registre dont un nom ressemble au nom donné."""
    liste = liste_a_jour()
    trouves = liste.rechercher(nom, seuil=seuil)
    return ResultatCriblage(
        nom_recherche=nom,
        seuil=seuil,
        date_registre=liste.date_publication,
        nb_correspondances=len(trouves),
        alerte=bool(trouves),
        correspondances=trouves,
    )