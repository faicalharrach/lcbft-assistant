"""Faits du dossier, calculés de façon déterministe à partir des transactions.

Le LLM ne calcule rien : il reçoit ces faits déjà chiffrés et les met en forme.
C'est ce qui permet ensuite de vérifier que la note ne contient pas de montant inventé.
"""

import pandas as pd

from src.detection.model import LIBELLES, REGLES
from src.generate.config import RAW_DIR
from src.generate.transactions import SEUIL_ALERTE_ESPECES

LIBELLES_REGLES = {r.code: r.libelle[0].lower() + r.libelle[1:] for r in REGLES}
SEUIL_COSI = 10_000  # art. L. 561-15-1 : espèces cumulées sur un mois calendaire

# Correspondance règles déclenchées -> fiches de typologie du corpus
TYPOLOGIES_PAR_REGLE = {
    "R1_FRACTIONNEMENT": "fractionnement",
    "R2_COMPTE_DE_PASSAGE": "compte_de_passage",
    "R3_PAYS_A_RISQUE": "pays_a_risque",
    "R4_INCOHERENCE_REVENUS": "incoherence_revenus",
    "R5_CRYPTO": "crypto",
    "R6_ESPECES": "fractionnement",
    "R7_BIENS_DE_VALEUR": "biens_valeur",
    "R8_SUR_DEPENSE_CSP": "incoherence_revenus",
}
TYPOLOGIES_PAR_VARIABLE = {
    "nb_depots_proches_seuil": "fractionnement", "part_especes_credits": "fractionnement",
    "nb_allers_retours_48h": "compte_de_passage", "montant_max_virement_entrant": "compte_de_passage",
    "nb_trans_pays_risque_eleve": "pays_a_risque", "montant_pays_risque_eleve": "pays_a_risque",
    "ratio_credits_revenu": "incoherence_revenus", "ratio_depenses_revenu": "incoherence_revenus",
    "nb_trans_crypto": "crypto", "montant_crypto": "crypto",
    "nb_trans_jeux": "jeux", "montant_jeux": "jeux",
    "montant_valeurs": "biens_valeur", "nb_trans_valeurs": "biens_valeur",
}
FICHE_PAR_TYPOLOGIE = {
    "fractionnement": "typo_fractionnement", "compte_de_passage": "typo_compte_de_passage",
    "pays_a_risque": "reg_pays_a_risque", "incoherence_revenus": "typo_incoherence_revenus",
    "crypto": "typo_crypto", "jeux": "typo_jeux", "biens_valeur": "typo_biens_valeur",
}
LIBELLES_TYPOLOGIES = {
    "fractionnement": "fractionnement des dépôts d'espèces",
    "compte_de_passage": "compte de passage",
    "pays_a_risque": "flux avec des pays à risque",
    "incoherence_revenus": "flux incohérents avec les revenus déclarés",
    "crypto": "crypto-actifs",
    "jeux": "jeux d'argent",
    "biens_valeur": "achats de biens de valeur",
}


def enrichir_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Ajoute marchand, famille MCC, pays et risque pays (mode ligne de commande)."""
    marchands = pd.read_csv(RAW_DIR / "ref_marchands.csv").rename(columns={"nom": "marchand"})
    mcc = pd.read_csv(RAW_DIR / "ref_mcc.csv")[["mcc", "famille"]]
    pays = pd.read_csv(RAW_DIR / "ref_pays.csv").rename(columns={"libelle": "pays", "niveau_risque": "risque_pays"})
    marchands = marchands.merge(mcc, on="mcc", how="left")[["marchand_id", "marchand", "famille"]]
    tx = tx.merge(marchands, on="marchand_id", how="left").merge(pays, on="code_pays", how="left")
    tx["date_trans"] = pd.to_datetime(tx["date_trans"])
    return tx


def _euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def _minuscule(texte: str) -> str:
    return texte[0].lower() + texte[1:] if texte else texte


def _ops(n: int) -> str:
    return f"{n} opération" + ("s" if n > 1 else "")


def _allers_retours(tx: pd.DataFrame, limite: int = 3) -> list:
    entrees = tx[(tx["sens"] == "credit") & (tx["type_trans"] == "VIREMENT") & (tx["montant"] >= 5_000)]
    sorties = tx[(tx["sens"] == "debit") & (tx["type_trans"] == "VIREMENT")]
    trouves = []
    for _, e in entrees.sort_values("date_trans").iterrows():
        fenetre = sorties[(sorties["date_trans"] > e["date_trans"])
                          & (sorties["date_trans"] <= e["date_trans"] + pd.Timedelta(hours=48))
                          & sorties["montant"].between(0.9 * e["montant"], e["montant"])]
        if not fenetre.empty:
            s = fenetre.iloc[0]
            heures = (s["date_trans"] - e["date_trans"]).total_seconds() / 3600
            trouves.append({
                "date_entree": e["date_trans"].strftime("%d/%m/%Y"),
                "montant_entree": round(float(e["montant"]), 2),
                "provenance": e.get("pays") or e["code_pays"],
                "montant_sortie": round(float(s["montant"]), 2),
                "destination": s.get("pays") or s["code_pays"],
                "delai_heures": round(heures),
            })
    return trouves[:limite]


def construire(client: dict, tx: pd.DataFrame) -> dict:
    """Faits chiffrés du dossier, prêts à être cités dans la note."""
    revenu_mensuel = float(client.get("salaire_declare", 0) + client.get("pension", 0))
    credits = tx[tx["sens"] == "credit"]["montant"].sum()
    debits = tx[tx["sens"] == "debit"]["montant"].sum()

    especes = tx[tx["type_trans"] == "DEPOT_ESPECES"].copy()
    especes["mois"] = especes["date_trans"].dt.to_period("M")
    especes_mois = especes.groupby("mois")["montant"].sum()
    sous_seuil = especes[especes["montant"].between(0.8 * SEUIL_ALERTE_ESPECES, SEUIL_ALERTE_ESPECES - 0.01)]

    risque = tx[tx["risque_pays"] == "eleve"]
    pays_risque = (risque.groupby("pays")["montant"].agg(["sum", "count"]).sort_values("sum", ascending=False))

    def famille(nom):
        f = tx[tx["famille"] == nom]
        return {"nb": int(len(f)), "montant": round(float(f["montant"].sum()), 2),
                "marchands": sorted(f["marchand"].dropna().unique().tolist())[:5]}

    regles = list(client.get("regles_declenchees") or [])
    raisons = list(client.get("raisons_modele") or [])
    typologies = []
    for code in regles:
        t = TYPOLOGIES_PAR_REGLE.get(code)
        if t and t not in typologies:
            typologies.append(t)
    for variable in raisons:
        t = TYPOLOGIES_PAR_VARIABLE.get(variable)
        if t and t not in typologies:
            typologies.append(t)

    faits = {
        "dossier": client["numero_dossier"],
        "client": f"{client.get('prenom', '')} {client.get('nom', '')}".strip(),
        "profil": f"{client.get('ccsp')}, {int(client.get('age', 0))} ans, client depuis "
                  f"{str(client.get('date_ouverture', ''))[:4]}",
        "revenu_mensuel_declare": round(revenu_mensuel, 2),
        "revenu_annuel_declare": round(12 * revenu_mensuel, 2),
        "score_final": float(client.get("score_final", 0)),
        "score_modele": float(client.get("score_if", 0)),
        "score_regles": float(client.get("score_regles", 0)),
        "rang": int(client.get("rang", 0)),
        "regles_declenchees": regles,
        "raisons_modele": raisons,
        "periode": f"{tx['date_trans'].min():%d/%m/%Y} au {tx['date_trans'].max():%d/%m/%Y}" if len(tx) else "",
        "nb_operations": int(len(tx)),
        "total_entrees": round(float(credits), 2),
        "total_sorties": round(float(debits), 2),
        "depots_especes": {
            "nb": int(len(especes)), "montant": round(float(especes["montant"].sum()), 2),
            "nb_sous_seuil_interne": int(len(sous_seuil)),
            "max_mensuel": round(float(especes_mois.max()), 2) if len(especes_mois) else 0.0,
            "mois_au_dessus_cosi": [str(m) for m, v in especes_mois.items() if v > SEUIL_COSI],
        },
        "allers_retours_48h": _allers_retours(tx),
        "pays_risque_eleve": [
            {"pays": p, "montant": round(float(r["sum"]), 2), "nb": int(r["count"])}
            for p, r in pays_risque.head(5).iterrows()
        ],
        "crypto": famille("crypto_change"),
        "jeux": famille("jeux"),
        "biens_valeur": famille("valeurs"),
        "typologies_candidates": typologies,
    }
    return faits


def montants_connus(faits: dict) -> set:
    """Tous les montants présents dans les faits, arrondis à l'euro."""
    valeurs = set()

    def parcourir(x):
        if isinstance(x, dict):
            for v in x.values():
                parcourir(v)
        elif isinstance(x, list):
            for v in x:
                parcourir(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            valeurs.add(round(float(x)))

    parcourir(faits)
    valeurs.update({SEUIL_COSI, SEUIL_ALERTE_ESPECES, round(0.8 * SEUIL_ALERTE_ESPECES), 5_000})
    return valeurs


def en_texte(faits: dict) -> str:
    """Version lisible des faits, injectée dans le prompt."""
    d = faits["depots_especes"]
    lignes = [
        f"Dossier {faits['dossier']} — {faits['client']} — {faits['profil']}.",
        f"Revenu déclaré : {_euros(faits['revenu_mensuel_declare'])} par mois, "
        f"soit {_euros(faits['revenu_annuel_declare'])} par an.",
        f"Période analysée : {faits['periode']} ({_ops(faits['nb_operations'])}).",
        f"Total des entrées : {_euros(faits['total_entrees'])}. Total des sorties : {_euros(faits['total_sorties'])}.",
        f"Score de risque : {faits['score_final']:.0f}/100 (modèle {faits['score_modele']:.0f}, "
        f"règles {faits['score_regles']:.0f}), rang {faits['rang']}.",
        f"Règles déclenchées : {'; '.join(LIBELLES_REGLES.get(r, r) for r in faits['regles_declenchees']) or 'aucune'}.",
        f"Ce qui pèse le plus dans le score du modèle : "
        f"{'; '.join(_minuscule(LIBELLES.get(v, v)) for v in faits['raisons_modele']) or 'rien de notable'}.",
    ]
    if d["nb"]:
        lignes.append(f"Dépôts d'espèces : {d['nb']} pour {_euros(d['montant'])}, dont {d['nb_sous_seuil_interne']} "
                      f"entre {_euros(0.8 * SEUIL_ALERTE_ESPECES)} et {_euros(SEUIL_ALERTE_ESPECES)} (seuil interne). "
                      f"Maximum mensuel : {_euros(d['max_mensuel'])}.")
    else:
        lignes.append("Aucun dépôt d'espèces sur la période.")
    if d["mois_au_dessus_cosi"]:
        lignes.append(f"Mois où les espèces cumulées dépassent {_euros(SEUIL_COSI)} : {', '.join(d['mois_au_dessus_cosi'])}.")
    for ar in faits["allers_retours_48h"]:
        lignes.append(f"Aller-retour : entrée de {_euros(ar['montant_entree'])} le {ar['date_entree']} "
                      f"({ar['provenance']}), sortie de {_euros(ar['montant_sortie'])} vers {ar['destination']} "
                      f"{ar['delai_heures']} h plus tard.")
    for p in faits["pays_risque_eleve"]:
        lignes.append(f"Pays à risque élevé : {p['pays']}, {_ops(p['nb'])}, {_euros(p['montant'])}.")
    for cle, nom in [("crypto", "Achats de crypto-actifs"), ("jeux", "Jeux d'argent"),
                     ("biens_valeur", "Achats de biens de valeur")]:
        f = faits[cle]
        if f["nb"]:
            lignes.append(f"{nom} : {_ops(f['nb'])} pour {_euros(f['montant'])} "
                          f"({', '.join(f['marchands'])}).")
    noms = [LIBELLES_TYPOLOGIES[t] for t in faits["typologies_candidates"]]
    lignes.append(f"Typologies candidates (issues des règles et du modèle) : {', '.join(noms) or 'aucune'}.")
    return "\n".join(lignes)
