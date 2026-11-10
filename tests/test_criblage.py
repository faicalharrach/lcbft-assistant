"""Tests de l'API de criblage : normalisation, score, recherche et fraîcheur de la copie.

Hors ligne : BigQuery est remplacé par une petite liste de test.
"""

import pytest
from fastapi.testclient import TestClient

from api import main
from api.criblage import ListeGels, normaliser, score_noms

NOMS = [
    {"id_registre": 4240, "type_nom": "principal", "nom_complet": "Grigory Vladimirovich SHILKIN"},
    {"id_registre": 4240, "type_nom": "alias", "nom_complet": "Григорий Владимирович ШИЛКИН"},
    {"id_registre": 10, "type_nom": "principal", "nom_complet": "ACME TRADING LLC"},
]
FICHES = {4240: {"nature": "Personne physique", "nom": "SHILKIN", "nationalites": [],
                 "dates_naissance": ["1976-10-20"], "fondements": ["(UE) 2022/332"]}}


def liste_test() -> ListeGels:
    return ListeGels(NOMS, FICHES, "2026-11-06")


@pytest.fixture
def client(monkeypatch):
    appels = []
    monkeypatch.setattr(ListeGels, "depuis_bigquery", classmethod(lambda cls: appels.append(1) or liste_test()))
    with TestClient(main.app) as c:
        c.appels = appels
        yield c


def test_normaliser():
    assert normaliser("  Shilkín-Jr. grigory ") == "SHILKIN JR GRIGORY"


@pytest.mark.parametrize("cherche, attendu_min, attendu_max", [
    ("Grigory Shilkin", 100, 100),     # patronyme en plus : pas de pénalité
    ("Shilkin Grigory", 100, 100),     # ordre des mots ignoré
    ("Gregory Shilkine", 85, 95),      # autre translittération
    ("Grigory Petrov", 0, 84),         # un prénom commun ne suffit pas
])
def test_score_noms(cherche, attendu_min, attendu_max):
    candidat = normaliser("Grigory Vladimirovich SHILKIN").split()
    assert attendu_min <= score_noms(normaliser(cherche).split(), candidat) <= attendu_max


def test_une_personne_une_seule_fois():
    """Trouvée par son nom principal ET son alias : une seule correspondance, la meilleure."""
    resultats = liste_test().rechercher("Grigory Shilkin")
    assert [r["id_registre"] for r in resultats] == [4240]


def test_criblage_alerte(client):
    r = client.get("/criblage", params={"nom": "Gregory Shilkine"})
    assert r.status_code == 200
    assert r.json()["alerte"] is True


def test_criblage_sans_alerte(client):
    r = client.get("/criblage", params={"nom": "Jean Dupont"})
    assert r.status_code == 200
    assert r.json()["alerte"] is False


def test_nom_trop_court_refuse(client):
    assert client.get("/criblage", params={"nom": "A"}).status_code == 422


def test_copie_recente_pas_rechargee(client):
    client.get("/sante")
    assert len(client.appels) == 1           # seulement le chargement du démarrage


def test_copie_perimee_rechargee(client):
    main.etat["charge_a"] -= main.DUREE_MAX_SECONDES + 1   # on « vieillit » la copie
    client.get("/sante")
    assert len(client.appels) == 2           # démarrage + rechargement