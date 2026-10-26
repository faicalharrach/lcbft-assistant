"""Agent de rédaction de la note d'analyse, orchestré avec LangGraph.

    collecter -> qualifier -> rechercher -> rediger -> verifier --(ok)--> fin
                                               ^           |
                                               +--(échec, 1 nouvel essai max)

- collecter  : faits chiffrés du dossier, calculés en Python (contexte.py) ;
- qualifier  : typologies candidates, déduites des règles et de SHAP ;
- rechercher : fiches réglementaires et typologiques pertinentes (rag.py) ;
- rediger    : Gemini rédige la note à partir des seuls faits et extraits fournis ;
- verifier   : garde-fous déterministes (sections, sources citées, articles de loi,
               montants, confidentialité). En cas d'échec, la note est réécrite
               une fois avec la liste des problèmes.

Le LLM met en forme et argumente ; il ne calcule pas et ne décide pas.
"""

import os
import re
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.agent import contexte, rag

SECTIONS = [
    "Synthèse", "Faits observés", "Typologies possibles", "Éléments à décharge",
    "Informations à demander au client", "Orientation proposée", "Références",
]
MAX_TENTATIVES = 2
MODELE_PAR_DEFAUT = "gemini-3.5-flash"

CONSIGNES = f"""Tu assistes un analyste conformité LCB-FT d'une banque de détail française.
Tu rédiges une note d'analyse d'alerte, en français, claire et professionnelle.

Règles impératives :
1. N'utilise QUE les faits du dossier et les extraits de la base de connaissances fournis.
   N'invente aucun montant, aucune date, aucun pays, aucun article de loi.
2. Recopie les montants tels qu'ils figurent dans les faits ; ne fais pas de nouveaux calculs.
3. Cite tes sources avec leur identifiant entre crochets, par exemple [reg_examen_renforce].
   Ne cite que les identifiants fournis.
4. Sépare les faits de leur interprétation. Présente aussi les explications légitimes possibles.
5. Tu ne conclus pas à la place de l'analyste : tu proposes une orientation
   (demande de justificatifs, classement motivé ou escalade au déclarant).
6. Les demandes au client restent dans le cadre de la connaissance client :
   ne mentionne jamais Tracfin, un soupçon ou une déclaration dans ce qui serait dit au client.

Structure obligatoire, avec ces titres de niveau 2 (##) et dans cet ordre :
{chr(10).join('## ' + s for s in SECTIONS)}
La section Références liste les identifiants cités et leur titre. Sois concis : 250 à 400 mots.
Réponds directement par la note en Markdown : pas de bloc de code, pas de préambule, pas de conclusion hors sections."""


class Etat(TypedDict, total=False):
    client: dict
    transactions: pd.DataFrame
    faits: dict
    faits_texte: str
    typologies: list
    documents: list
    note: str
    problemes: list
    tentatives: int
    brouillons: list


# --------------------------------------------------------------------------
# Modèle de langage
# --------------------------------------------------------------------------
class LLMSimule:
    """Rédacteur déterministe, sans appel réseau : tests et démonstration hors ligne."""

    def invoke(self, messages):
        faits, documents = messages[-1]["faits"], messages[-1]["documents"]
        lignes_faits = contexte.en_texte(faits).splitlines()
        typos = [contexte.LIBELLES_TYPOLOGIES[t] for t in faits["typologies_candidates"]] or ["aucune identifiée"]
        ids = [d.id for d in documents]
        ref_typo = next((i for i in ids if i.startswith("typo_")), ids[0])
        texte = "\n".join([
            "## Synthèse",
            f"Le dossier {faits['dossier']} présente un score de risque de {faits['score_final']:.0f}/100. "
            f"Les opérations observées appellent un examen renforcé [reg_examen_renforce].",
            "## Faits observés",
            *[f"- {ligne}" for ligne in lignes_faits[2:-1]],
            "## Typologies possibles",
            f"- {', '.join(typos).capitalize()} [{ref_typo}].",
            "## Éléments à décharge",
            "- Les écarts observés peuvent avoir une explication légitime (vente d'un bien, héritage, "
            f"changement de situation) qu'il convient de documenter [{ref_typo}].",
            "## Informations à demander au client",
            "- Origine et destination des fonds, avec justificatifs [reg_examen_renforce].",
            "- Mise à jour de la situation professionnelle et des revenus [reg_vigilance_constante].",
            "## Orientation proposée",
            "Demander les justificatifs puis réexaminer le dossier ; escalader au déclarant si le doute "
            "n'est pas levé [reg_declaration_soupcon].",
            "## Références",
            *[f"- [{d.id}] {d.titre}" for d in documents],
        ])
        return type("Reponse", (), {"content": texte})()


def creer_llm():
    if os.environ.get("AGENT_LLM", "vertex") == "simule":
        return LLMSimule()
    from langchain_google_vertexai import ChatVertexAI

    # gemini-2.5-flash a été retiré de Vertex AI le 20/10/2026 : modèle et région restent configurables.
    return ChatVertexAI(
        model=os.environ.get("GEMINI_MODEL", MODELE_PAR_DEFAUT),
        location=os.environ.get("VERTEX_LOCATION", "global"),
        temperature=0.2,
        # Les modèles Gemini 3 « réfléchissent » avant de répondre et cette réflexion consomme
        # le budget de sortie : avec 2 048 tokens, la note était tronquée.
        max_output_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "8192")),
    )


# --------------------------------------------------------------------------
# Nœuds du graphe
# --------------------------------------------------------------------------
def collecter(etat: Etat) -> Etat:
    faits = contexte.construire(etat["client"], etat["transactions"])
    return {"faits": faits, "faits_texte": contexte.en_texte(faits), "tentatives": 0, "problemes": []}


def qualifier(etat: Etat) -> Etat:
    return {"typologies": etat["faits"]["typologies_candidates"]}


def rechercher(etat: Etat) -> Etat:
    """Recherche hybride : fiches imposées par les typologies détectées, complétées par
    les fiches sémantiquement les plus proches de la description du dossier."""
    tous = {d.id: d for d in rag.charger_corpus()}
    # Toujours présents : le cadre de l'analyse et la méthode de rédaction.
    retenus = ["reg_examen_renforce", "reg_vigilance_constante", "reg_declaration_soupcon",
               "reg_confidentialite", "methode_note"]
    for typologie in etat["typologies"]:
        fiche = contexte.FICHE_PAR_TYPOLOGIE.get(typologie)
        if fiche and fiche not in retenus:
            retenus.append(fiche)
    if etat["faits"]["depots_especes"]["nb"] and "reg_especes_cosi" not in retenus:
        retenus.append("reg_especes_cosi")
    for doc in rag.moteur().chercher(etat["faits_texte"], k=3):
        if doc.id not in retenus:
            retenus.append(doc.id)
    return {"documents": [tous[i] for i in retenus]}


def rediger(etat: Etat) -> Etat:
    extraits = "\n\n".join(d.pour_prompt() for d in etat["documents"])
    demande = f"FAITS DU DOSSIER\n{etat['faits_texte']}\n\nBASE DE CONNAISSANCES\n{extraits}"
    if etat.get("problemes"):
        demande += ("\n\nTA VERSION PRÉCÉDENTE A ÉTÉ REJETÉE PAR LES CONTRÔLES. Corrige ces points :\n- "
                    + "\n- ".join(etat["problemes"]))
    llm = creer_llm()
    if isinstance(llm, LLMSimule):
        reponse = llm.invoke([{"faits": etat["faits"], "documents": etat["documents"]}])
    else:
        reponse = llm.invoke([("system", CONSIGNES), ("human", demande)])
    brut = _texte_reponse(reponse)
    meta = getattr(reponse, "response_metadata", None) or {}
    brouillon = {
        "tentative": etat.get("tentatives", 0) + 1,
        "fin": str(meta.get("finish_reason", "")),
        "usage": {k: v for k, v in (getattr(reponse, "usage_metadata", None) or {}).items()
                  if isinstance(v, (int, float))},
        "texte": brut,
    }
    return {"note": normaliser_note(brut), "tentatives": brouillon["tentative"],
            "brouillons": etat.get("brouillons", []) + [brouillon]}


def _texte_reponse(reponse) -> str:
    """Texte de la réponse, sans les éventuelles parties de « réflexion » du modèle."""
    contenu = reponse.content
    if isinstance(contenu, str):
        return contenu
    morceaux = []
    for partie in contenu:
        if isinstance(partie, str):
            morceaux.append(partie)
        elif isinstance(partie, dict) and partie.get("type", "text") == "text":
            morceaux.append(partie.get("text", ""))
    return "".join(morceaux)


def _sans_accents(texte: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn").lower()


_TITRES = {_sans_accents(s): s for s in SECTIONS}


def normaliser_note(texte: str) -> str:
    """Remet la note au format attendu quand le modèle s'en écarte légèrement :
    bloc de code autour de la note, titres en gras, en # ou ###, suivis de « : »."""
    texte = texte.strip()
    texte = re.sub(r"^```[a-zA-Z]*\s*\n", "", texte)
    texte = re.sub(r"\n```\s*$", "", texte)
    lignes = []
    for ligne in texte.splitlines():
        candidat = re.sub(r"^\s*#{1,6}\s*|\*\*|__", "", ligne).strip().rstrip(":").strip()
        candidat = re.sub(r"^\d+[.)]\s*", "", candidat)
        titre = _TITRES.get(_sans_accents(candidat))
        lignes.append(f"## {titre}" if titre else ligne)
    return "\n".join(lignes).strip()


# --------------------------------------------------------------------------
# Garde-fous
# --------------------------------------------------------------------------
_MONTANT = re.compile(r"(\d{1,3}(?:[   .]\d{3})+|\d+)(?:,\d+)?\s?(?:€|euros?)")
_ARTICLE = re.compile(r"L\.?\s?561-\d+(?:-\d+)?")
_CITATION = re.compile(r"\[([a-z_]+)\]")


def _normaliser_article(a: str) -> str:
    return re.sub(r"[\s.]", "", a)


def controler(note: str, faits: dict, documents: list) -> list:
    problemes = []
    for section in SECTIONS:
        if not re.search(rf"^##\s+{re.escape(section)}\s*$", note, flags=re.MULTILINE):
            problemes.append(f"Section manquante ou mal titrée : « ## {section} ».")

    ids = {d.id for d in documents}
    citees = set(_CITATION.findall(note))
    if not citees:
        problemes.append("Aucune source citée entre crochets.")
    for inconnue in sorted(citees - ids):
        problemes.append(f"Source citée inexistante : [{inconnue}].")

    textes_sources = _normaliser_article(" ".join(f"{d.source} {d.texte}" for d in documents))
    for article in sorted(set(_ARTICLE.findall(note))):
        if _normaliser_article(article) not in textes_sources:
            problemes.append(f"Article {article} absent des extraits fournis : ne pas le citer.")

    connus = contexte.montants_connus(faits)
    for brut in _MONTANT.findall(note):
        valeur = int(re.sub(r"[   .]", "", brut))
        if valeur >= 100 and not any(abs(valeur - c) <= max(1, 0.005 * c) for c in connus):
            problemes.append(f"Montant {brut} € absent des faits du dossier.")

    demande = re.search(r"##\s+Informations à demander au client(.*?)(?=^##\s|\Z)", note, flags=re.S | re.M)
    if demande and re.search(r"tracfin|soupçon|déclaration", demande.group(1), flags=re.I):
        problemes.append("La section des demandes au client mentionne Tracfin, un soupçon ou une "
                         "déclaration : risque de divulgation (art. L. 561-18).")
    return problemes


def verifier(etat: Etat) -> Etat:
    return {"problemes": controler(etat["note"], etat["faits"], etat["documents"])}


def apres_verification(etat: Etat) -> str:
    if etat["problemes"] and etat["tentatives"] < MAX_TENTATIVES:
        return "rediger"
    return END


def construire_graphe():
    graphe = StateGraph(Etat)
    for nom, fonction in [("collecter", collecter), ("qualifier", qualifier), ("rechercher", rechercher),
                          ("rediger", rediger), ("verifier", verifier)]:
        graphe.add_node(nom, fonction)
    graphe.set_entry_point("collecter")
    graphe.add_edge("collecter", "qualifier")
    graphe.add_edge("qualifier", "rechercher")
    graphe.add_edge("rechercher", "rediger")
    graphe.add_edge("rediger", "verifier")
    graphe.add_conditional_edges("verifier", apres_verification, {"rediger": "rediger", END: END})
    return graphe.compile()


def generer_note(client: dict, transactions: pd.DataFrame) -> dict:
    """Point d'entrée : renvoie la note, les sources, les contrôles et le nombre d'essais."""
    resultat = construire_graphe().invoke({"client": client, "transactions": transactions})
    sortie = {
        "note": resultat["note"],
        "documents": resultat["documents"],
        "problemes": resultat["problemes"],
        "tentatives": resultat["tentatives"],
        "faits": resultat["faits"],
        "brouillons": resultat.get("brouillons", []),
    }
    _journaliser(client.get("numero_dossier", "inconnu"), sortie)
    return sortie


def _journaliser(dossier: str, sortie: dict) -> None:
    """Garde la trace de chaque génération (réponses brutes, contrôles) pour le diagnostic."""
    import json
    from datetime import datetime

    from src.generate.config import PROJECT_ROOT

    dossier_logs = PROJECT_ROOT / "data" / "agent_logs"
    dossier_logs.mkdir(parents=True, exist_ok=True)
    (dossier_logs / f"{dossier}.json").write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "modele": "simule" if os.environ.get("AGENT_LLM", "vertex") == "simule"
        else os.environ.get("GEMINI_MODEL", MODELE_PAR_DEFAUT),
        "problemes": sortie["problemes"],
        "brouillons": sortie["brouillons"],
        "sources": [d.id for d in sortie["documents"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
