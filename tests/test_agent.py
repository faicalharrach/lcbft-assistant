"""Tests de l'agent : base de connaissances, garde-fous et boucle de réécriture.

Exécutés hors ligne : rédacteur simulé et recherche TF-IDF (aucun appel à Vertex AI).
"""

import pytest

from src.agent import graphe, rag
from src.agent.graphe import SECTIONS, controler
from src.generate.config import RAW_DIR


@pytest.fixture(autouse=True)
def hors_ligne(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "simule")
    monkeypatch.setenv("RAG_MOTEUR", "local")
    rag.moteur.cache_clear()


@pytest.fixture(scope="module")
def dossier():
    if not (RAW_DIR / "transactions.csv").exists():
        pytest.skip("Lancer d'abord python -m src.generate.run_all et python -m src.detection.run --source local")
    from src.agent.run import charger_dossier

    return charger_dossier("D001407")


def _note_valide(ids):
    corps = {s: f"Texte [{ids[0]}]." for s in SECTIONS}
    return "\n".join(f"## {s}\n{t}" for s, t in corps.items())


def test_corpus_complet():
    documents = rag.charger_corpus()
    assert len(documents) >= 10
    assert len({d.id for d in documents}) == len(documents)
    assert all(d.source and d.texte for d in documents)


def test_recherche_locale_pertinente():
    resultats = rag.moteur("local").chercher("dépôts d'espèces répétés juste sous un seuil", k=3)
    assert "typo_fractionnement" in [d.id for d in resultats]


def test_note_valide_acceptee():
    docs = rag.charger_corpus()[:3]
    assert controler(_note_valide([docs[0].id]), {"total": 1234.0}, docs) == []


@pytest.mark.parametrize("ajout, attendu", [
    ("Montant viré : 98 765 €.", "Montant"),
    ("Voir l'article L. 561-99.", "Article"),
    ("Voir [fiche_inventee].", "inexistante"),
])
def test_garde_fous_detectent_les_inventions(ajout, attendu):
    docs = rag.charger_corpus()[:3]
    note = _note_valide([docs[0].id]) + "\n" + ajout
    assert any(attendu in p for p in controler(note, {"total": 1234.0}, docs))


def test_section_manquante_detectee():
    docs = rag.charger_corpus()[:3]
    note = _note_valide([docs[0].id]).replace("## Éléments à décharge", "## Autre")
    assert any("Éléments à décharge" in p for p in controler(note, {}, docs))


def test_divulgation_au_client_detectee():
    docs = rag.charger_corpus()[:3]
    note = _note_valide([docs[0].id]).replace(
        "## Informations à demander au client\n",
        "## Informations à demander au client\nPrévenir le client de la déclaration à Tracfin.\n")
    assert any("L. 561-18" in p for p in controler(note, {}, docs))


def test_agent_de_bout_en_bout(dossier):
    client, tx = dossier
    resultat = graphe.generer_note(client, tx)
    assert resultat["problemes"] == []
    assert resultat["tentatives"] == 1
    assert "compte_de_passage" in resultat["faits"]["typologies_candidates"]
    assert "typo_compte_de_passage" in [d.id for d in resultat["documents"]]


def test_note_rejetee_puis_reecrite(dossier, monkeypatch):
    """Première version invalide (montant inventé) : le graphe doit relancer la rédaction."""
    client, tx = dossier
    appels = []

    class LLMFautif(graphe.LLMSimule):
        def invoke(self, messages):
            reponse = super().invoke(messages)
            appels.append(1)
            if len(appels) == 1:
                reponse.content += "\nMontant total suspect : 123 456 €."
            return reponse

    monkeypatch.setattr(graphe, "creer_llm", lambda: LLMFautif())
    resultat = graphe.generer_note(client, tx)
    assert resultat["tentatives"] == 2
    assert resultat["problemes"] == []
    assert "123 456" not in resultat["note"]


@pytest.mark.parametrize("variante", [
    "```markdown\n{note}\n```",
    "{note_gras}",
    "{note_dieses}",
])
def test_normalisation_des_titres(variante):
    """Les écarts de format courants des LLM sont corrigés avant les contrôles."""
    from src.agent.graphe import normaliser_note

    sections = graphe.SECTIONS
    note = "\n".join(f"## {s}\nTexte." for s in sections)
    note_gras = "\n".join(f"**{s} :**\nTexte." for s in sections)
    note_dieses = "\n".join(f"### {i + 1}. {s}\nTexte." for i, s in enumerate(sections))
    brut = variante.format(note=note, note_gras=note_gras, note_dieses=note_dieses)
    resultat = normaliser_note(brut)
    for s in sections:
        assert f"## {s}" in resultat.splitlines()
    assert "```" not in resultat


def test_parties_de_reflexion_ignorees():
    reponse = type("R", (), {"content": [{"type": "thinking", "thinking": "brouillon interne"},
                                         {"type": "text", "text": "## Synthèse\nOK"}]})()
    assert graphe._texte_reponse(reponse) == "## Synthèse\nOK"
