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


def test_note_archivee_puis_relue(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "CHEMIN_NOTES", tmp_path / "notes.jsonl")
    journal.derniere_note.clear()
    doc = type("Doc", (), {"id": "reg_examen_renforce"})()
    resultat = {"note": "## Synthèse\nV1", "documents": [doc], "problemes": [], "tentatives": 1,
                "brouillons": [{"usage": {"input_tokens": 6000, "output_tokens": 2500}}]}
    journal.enregistrer_note("D000001", "gemini-3.5-flash", resultat)
    journal.enregistrer_note("D000001", "gemini-3.5-flash", resultat | {"note": "## Synthèse\nV2"})

    note = journal.derniere_note("D000001")
    assert note["note"].endswith("V2")
    assert note["sources"] == ["reg_examen_renforce"]
    assert note["tokens_entree"] == 6000
    assert journal.derniere_note("D999999") is None
