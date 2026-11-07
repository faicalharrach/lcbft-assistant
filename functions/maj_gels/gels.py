"""Lecture et mise à plat du registre national des gels (format JSON de la DG Trésor).

Test local :
    python functions/maj_gels/gels.py data/gels/registre.json
"""

import json
import sys
from collections import Counter


def _valeurs(entree: dict, type_champ: str) -> list[dict]:
    """Toutes les valeurs d'un TypeChamp donné (liste vide s'il est absent)."""
    valeurs = []
    for detail in entree.get("RegistreDetail", []):
        if detail.get("TypeChamp") == type_champ:
            valeurs.extend(detail.get("Valeur", []))
    return valeurs


def _date(v: dict) -> str:
    """{'Jour': '20', 'Mois': '10', 'Annee': '1976'} -> '1976-10-20' ; parties manquantes omises."""
    parties = [v.get("Annee", ""), v.get("Mois", "").zfill(2) if v.get("Mois") else "",
               v.get("Jour", "").zfill(2) if v.get("Jour") else ""]
    return "-".join(p for p in parties if p)


def aplatir(entree: dict) -> dict:
    """Une entrée du registre -> une ligne plate, prête pour BigQuery."""
    identifiants = (
        [f"PASSEPORT {v['NumeroPasseport'].strip()}"
         for v in _valeurs(entree, "PASSEPORT") if v.get("NumeroPasseport", "").strip()]
        + [f"{v.get('Commentaire') or 'AUTRE'} {v['NumeroCarte'].strip()}"
           for v in _valeurs(entree, "AUTRE_IDENTITE") if v.get("NumeroCarte", "").strip()]
    )
    return {
        "id_registre": entree["IdRegistre"],
        "nature": entree.get("Nature"),
        "nom": entree.get("Nom"),
        "prenoms": [v["Prenom"] for v in _valeurs(entree, "PRENOM") if v.get("Prenom")],
        "alias": [v["Alias"] for v in _valeurs(entree, "ALIAS") if v.get("Alias")],
        "dates_naissance": [_date(v) for v in _valeurs(entree, "DATE_DE_NAISSANCE")],
        "lieux_naissance": [", ".join(filter(None, [v.get("Lieu"), v.get("Pays")]))
                            for v in _valeurs(entree, "LIEU_DE_NAISSANCE")],
        "nationalites": [v["Pays"] for v in _valeurs(entree, "NATIONALITE") if v.get("Pays")],
        "sexe": next((v.get("Sexe") for v in _valeurs(entree, "SEXE")), None),
        "identifiants": identifiants,
        "numero_omi": next((v.get("NumeroOMI") for v in _valeurs(entree, "NUMERO_OMI")), None),
        "fondements": [v.get("FondementJuridiqueLabel") for v in _valeurs(entree, "FONDEMENT_JURIDIQUE")],
        "motifs": " ".join(v.get("Motifs", "") for v in _valeurs(entree, "MOTIFS")).strip() or None,
        # tout le détail d'origine, pour l'audit : rien n'est perdu à la mise à plat
        "details_json": json.dumps(entree.get("RegistreDetail", []), ensure_ascii=False),
    }

def noms_a_cribler(ligne: dict) -> list[dict]:
    """Tous les noms sous lesquels une personne peut apparaître : nom principal et alias."""
    noms = []
    if ligne["prenoms"]:
        noms += [{"id_registre": ligne["id_registre"], "type_nom": "principal",
                  "nom_complet": f"{p} {ligne['nom']}".strip()} for p in ligne["prenoms"]]
    elif ligne["nom"]:
        noms.append({"id_registre": ligne["id_registre"], "type_nom": "principal", "nom_complet": ligne["nom"]})
    noms += [{"id_registre": ligne["id_registre"], "type_nom": "alias", "nom_complet": a} for a in ligne["alias"]]
    return noms


def lire(contenu: dict) -> tuple[str, list[dict], list[dict]]:
    """Registre complet -> (date de publication, entrées à plat, noms à cribler)."""
    publications = contenu["Publications"]
    lignes = [aplatir(e) for e in publications["PublicationDetail"]]
    noms = [n for ligne in lignes for n in noms_a_cribler(ligne)]
    return publications["DatePublication"], lignes, noms


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        brut = json.load(f)
    date_pub, lignes, noms = lire(brut)
    types = Counter(d["TypeChamp"] for e in brut["Publications"]["PublicationDetail"]
                    for d in e.get("RegistreDetail", []))
    print(f"Publication : {date_pub}")
    print(f"Entrées     : {len(lignes):,}")
    print(f"Noms        : {len(noms):,} (principaux + alias)\n")
    print("Nature :", dict(Counter(l["nature"] for l in lignes)), "\n")
    print("Types de champs présents :")
    for t, n in types.most_common():
        print(f"  {t:<25} {n:>6}")
    print("\nExemple :", json.dumps({k: v for k, v in lignes[0].items() if k != "details_json"},
                                    ensure_ascii=False, indent=2))