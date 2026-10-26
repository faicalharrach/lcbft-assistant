"""Tests du journal des décisions (stockage local)."""

import pytest

from src.app import journal


@pytest.fixture(autouse=True)
def journal_temporaire(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "BACKEND", "local")
    monkeypatch.setattr(journal, "CHEMIN_LOCAL", tmp_path / "decisions.csv")
    journal.lire.clear()


def test_journal_vide():
    assert journal.lire().empty
    assert journal.statuts() == {}


def test_decisions_ajoutees_jamais_ecrasees():
    journal.enregistrer("D000001", "En cours d'analyse", "Justificatifs demandés.", "Alice", 82.0, True)
    journal.enregistrer("D000001", "Classé sans suite", "Acte de succession reçu.", "Alice", 82.0, True)
    journal.enregistrer("D000002", "Escaladé (déclaration de soupçon)", "Origine des fonds non justifiée.", "", 95.0, False)

    assert len(journal.lire()) == 3
    assert len(journal.historique("D000001")) == 2
    assert journal.statuts() == {"D000001": "Classé sans suite", "D000002": "Escaladé (déclaration de soupçon)"}
    assert journal.historique("D000002")["analyste"].iloc[0] == "analyste"
