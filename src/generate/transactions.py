"""Table transactions : activité normale sur 2025, puis injection de 5 typologies.

Sorties :
- transactions.csv         : toutes les opérations, SANS étiquette (ce que voit le modèle) ;
- labels_clients.csv       : clients porteurs d'une typologie (vérité terrain) ;
- labels_transactions.csv  : opérations injectées et leur typologie (pour l'interface
                             et l'évaluation, jamais pour l'entraînement) ;
- profils_legitimes.csv    : clients légitimes au comportement atypique ("faux amis"),
                             pour analyser les faux positifs.

Séparer les étiquettes des transactions évite toute fuite de la vérité terrain
dans les features.

Activité normale d'un client, chaque mois :
- un virement entrant (salaire et/ou pension, ou aides si pas de revenu déclaré) ;
- un virement sortant de charges (loyer, crédit...) ;
- des achats carte chez des marchands courants, des retraits DAB ;
- quelques comportements "à risque" mais légitimes : paris occasionnels,
  petits achats crypto, un bijou, des envois d'argent à la famille, des dépôts
  d'espèces (surtout pour les artisans et commerçants).
Ces cas légitimes sont indispensables : sans eux, le modèle apprendrait que
"MCC jeux" ou "crypto" veut dire "suspect", et les faux positifs exploseraient.
"""

import numpy as np
import pandas as pd

from src.generate.config import DATE_DEBUT, get_rng

# --- Paramètres de l'activité normale -------------------------------------
REVENU_SANS_ACTIVITE = 550      # aides / soutien familial, en €/mois
PART_BUDGET_ACHATS = 0.45       # part du revenu dépensée en carte
PART_CHARGES = 0.30             # loyer, crédits, abonnements
PART_JOUEURS = 0.10             # clients qui parient de temps en temps
PART_CRYPTO_LEGITIME = 0.06     # clients qui achètent un peu de crypto
PART_ACHAT_VALEURS = 0.05       # clients qui achètent un bijou dans l'année
PART_ENVOIS_FAMILLE = 0.08      # clients qui envoient de l'argent à l'étranger

# --- Paramètres des typologies --------------------------------------------
N_PAR_TYPOLOGIE = 30            # 5 x 30 = 150 clients suspects, soit 3 %
SEUIL_ALERTE_ESPECES = 3_000    # seuil interne FICTIF que le fractionnement contourne
TYPOLOGIES = [
    "fractionnement",
    "compte_de_passage",
    "pays_a_risque",
    "incoherence_revenus",
    "crypto",
]

COLONNES = [
    "numero_dossier", "date_trans", "montant", "sens", "type_trans",
    "marchand_id", "code_pays",
]


# ==========================================================================
# Outils
# ==========================================================================
def _lognormal_moyenne(rng, moyenne, sigma, size=None):
    """Log-normale paramétrée par sa MOYENNE (et non sa médiane)."""
    mu = np.log(np.maximum(moyenne, 1)) - sigma**2 / 2
    return rng.lognormal(mu, sigma, size)


def _date(rng, debut: pd.Timestamp, jours_max: int = 365) -> pd.Timestamp:
    """Date aléatoire entre debut et debut + jours_max, en journée (7h-22h)."""
    fin_annee = pd.Timestamp(DATE_DEBUT) + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    jours_max = max(min(jours_max, (fin_annee - debut).days), 0)
    jour = int(rng.integers(0, jours_max + 1))
    seconde = int(rng.integers(7 * 3600, 22 * 3600))
    return debut + pd.Timedelta(days=jour, seconds=seconde)


def _op(dossier, date, montant, sens, type_trans, pays="FR", marchand=None):
    return {
        "numero_dossier": dossier,
        "date_trans": date,
        "montant": round(float(montant), 2),
        "sens": sens,
        "type_trans": type_trans,
        "marchand_id": marchand,
        "code_pays": pays,
    }


class _Contexte:
    """Référentiels pré-découpés, utilisés par toutes les fonctions."""

    def __init__(self, marchands, mcc, pays):
        m = marchands.merge(mcc[["mcc", "famille", "niveau_risque"]], on="mcc")
        self.marchands = m.set_index("marchand_id")
        courants = m[m["niveau_risque"] == "faible"]
        poids = np.where(courants["code_pays"] == "FR", 1.0, 0.1)
        self.courants_ids = courants["marchand_id"].to_numpy()
        self.courants_poids = poids / poids.sum()
        self.par_famille = {
            f: g["marchand_id"].to_numpy() for f, g in m.groupby("famille")
        }
        self.pays_par_risque = {
            r: g["code_pays"].to_numpy() for r, g in pays.groupby("niveau_risque")
        }
        self.pays_etrangers_surs = np.array(
            [p for p in self.pays_par_risque["faible"] if p != "FR"]
        )

    def marchand(self, rng, famille):
        mid = str(rng.choice(self.par_famille[famille]))
        return mid, self.marchands.at[mid, "code_pays"]

    def pays(self, rng, risque):
        if risque == "faible":
            return str(rng.choice(self.pays_etrangers_surs))
        return str(rng.choice(self.pays_par_risque[risque]))


# ==========================================================================
# Activité normale (vectorisée pour les gros volumes)
# ==========================================================================
def _revenu_effectif(clients: pd.DataFrame) -> np.ndarray:
    revenu = (clients["salaire_declare"] + clients["pension"]).to_numpy(dtype=float)
    return np.where(revenu > 0, revenu, REVENU_SANS_ACTIVITE)


def _evenements_mensuels(rng, n_clients, intensite):
    """Tire un nombre d'événements par client et par mois (loi de Poisson).

    Renvoie, pour chaque événement : l'indice du client et le mois (0-11).
    """
    comptes = rng.poisson(np.repeat(intensite[:, None], 12, axis=1))
    cellule = np.repeat(np.arange(n_clients * 12), comptes.ravel())
    return cellule // 12, cellule % 12


def _dates_dans_le_mois(rng, mois, jour_min=0, jour_max=None):
    """Date aléatoire dans le mois donné, entre le jour jour_min et jour_max (exclu)."""
    debuts = pd.date_range(DATE_DEBUT, periods=12, freq="MS")
    nb_jours = debuts.days_in_month.to_numpy()
    fin = nb_jours[mois] if jour_max is None else np.minimum(jour_max, nb_jours[mois])
    jours = rng.integers(jour_min, fin)
    secondes = rng.integers(7 * 3600, 22 * 3600, size=len(mois))
    return (
        debuts[mois]
        + pd.to_timedelta(jours, unit="D")
        + pd.to_timedelta(secondes, unit="s")
    )


def _activite_normale(rng, clients: pd.DataFrame, ctx: _Contexte) -> pd.DataFrame:
    n = len(clients)
    dossiers = clients["numero_dossier"].to_numpy()
    revenu = _revenu_effectif(clients)
    blocs = []

    # 1. Virements entrants mensuels (début de mois)
    c, m = np.repeat(np.arange(n), 12), np.tile(np.arange(12), n)
    blocs.append(pd.DataFrame({
        "numero_dossier": dossiers[c],
        "date_trans": _dates_dans_le_mois(rng, m, 0, 4),
        "montant": np.round(revenu[c] * rng.normal(1, 0.02, len(c)), 2),
        "sens": "credit", "type_trans": "VIREMENT", "marchand_id": None, "code_pays": "FR",
    }))

    # 2. Charges mensuelles (loyer, crédits)
    blocs.append(pd.DataFrame({
        "numero_dossier": dossiers[c],
        "date_trans": _dates_dans_le_mois(rng, m, 3, 9),
        "montant": np.round(revenu[c] * PART_CHARGES * rng.normal(1, 0.01, len(c)), 2),
        "sens": "debit", "type_trans": "VIREMENT", "marchand_id": None, "code_pays": "FR",
    }))

    # 3. Achats carte chez des marchands courants
    lam = np.clip(5 + revenu / 400, 5, 20)
    c, m = _evenements_mensuels(rng, n, lam)
    moyenne = revenu[c] * PART_BUDGET_ACHATS / lam[c]
    marchands = rng.choice(ctx.courants_ids, size=len(c), p=ctx.courants_poids)
    blocs.append(pd.DataFrame({
        "numero_dossier": dossiers[c],
        "date_trans": _dates_dans_le_mois(rng, m),
        "montant": np.round(_lognormal_moyenne(rng, moyenne, 0.9), 2),
        "sens": "debit", "type_trans": "ACHAT", "marchand_id": marchands,
        "code_pays": ctx.marchands.loc[marchands, "code_pays"].to_numpy(),
    }))

    # 4. Retraits DAB (multiples de 10, majoritairement en France)
    c, m = _evenements_mensuels(rng, n, np.full(n, 1.2))
    a_l_etranger = rng.random(len(c)) < 0.02
    pays_dab = np.where(a_l_etranger, rng.choice(ctx.pays_etrangers_surs, len(c)), "FR")
    blocs.append(pd.DataFrame({
        "numero_dossier": dossiers[c],
        "date_trans": _dates_dans_le_mois(rng, m),
        "montant": rng.choice([20, 40, 50, 60, 80, 100, 150, 200], len(c),
                              p=[.15, .2, .2, .15, .1, .1, .05, .05]).astype(float),
        "sens": "debit", "type_trans": "DAB", "marchand_id": None, "code_pays": pays_dab,
    }))

    # 5. Dépôts d'espèces légitimes (fréquents chez les artisans et commerçants)
    artisan = clients["ccsp"].str.startswith("Artisans").to_numpy()
    c, m = _evenements_mensuels(rng, n, np.where(artisan, 0.6, 0.03))
    moyenne = np.where(artisan[c], 900, 150)
    blocs.append(pd.DataFrame({
        "numero_dossier": dossiers[c],
        "date_trans": _dates_dans_le_mois(rng, m),
        "montant": np.round(_lognormal_moyenne(rng, moyenne, 0.6), -1).clip(min=10),
        "sens": "credit", "type_trans": "DEPOT_ESPECES", "marchand_id": None, "code_pays": "FR",
    }))

    # 6. Comportements "à risque" mais légitimes (petits volumes, boucle simple)
    ops = []
    for i in np.flatnonzero(rng.random(n) < PART_JOUEURS):
        for _ in range(rng.poisson(6)):
            mid, pays = ctx.marchand(rng, "jeux")
            date = _date(rng, pd.Timestamp(DATE_DEBUT))
            ops.append(_op(dossiers[i], date, _lognormal_moyenne(rng, 20, 0.7), "debit", "ACHAT", pays, mid))
    for i in np.flatnonzero(rng.random(n) < PART_CRYPTO_LEGITIME):
        for _ in range(1 + rng.poisson(2)):
            mid, pays = ctx.marchand(rng, "crypto_change")
            date = _date(rng, pd.Timestamp(DATE_DEBUT))
            ops.append(_op(dossiers[i], date, _lognormal_moyenne(rng, 100, 0.6), "debit", "ACHAT", pays, mid))
    for i in np.flatnonzero(rng.random(n) < PART_ACHAT_VALEURS):
        mid, pays = ctx.marchand(rng, "valeurs")
        date = _date(rng, pd.Timestamp(DATE_DEBUT))
        ops.append(_op(dossiers[i], date, _lognormal_moyenne(rng, 300, 0.6), "debit", "ACHAT", pays, mid))
    for i in np.flatnonzero(rng.random(n) < PART_ENVOIS_FAMILLE):
        for _ in range(1 + rng.poisson(2)):
            # 5 % des envois partent vers un pays à risque élevé : bruit légitime
            risque = "eleve" if rng.random() < 0.05 else str(rng.choice(["faible", "moyen"], p=[.7, .3]))
            date = _date(rng, pd.Timestamp(DATE_DEBUT))
            ops.append(_op(dossiers[i], date, _lognormal_moyenne(rng, 300, 0.6), "debit",
                           "VIREMENT", ctx.pays(rng, risque)))
    blocs.append(pd.DataFrame(ops, columns=COLONNES))

    return pd.concat(blocs, ignore_index=True)


# ==========================================================================
# Faux amis : clients légitimes au comportement atypique
# ==========================================================================
# Sans eux, tout client atypique serait suspect et l'évaluation serait trop
# optimiste. En vrai, ce sont eux qui génèrent les faux positifs.
PART_HERITAGE_REINVESTI = 0.01   # grosse entrée (héritage, vente) vite reversée au notaire
PART_FAMILLE_PAYS_RISQUE = 0.01  # envois mensuels modestes à la famille dans un pays à risque
PART_GROS_INVESTISSEUR = 0.005   # hauts revenus qui investissent beaucoup en crypto


def _faux_amis(rng, clients: pd.DataFrame, ctx: _Contexte, exclus: set) -> tuple:
    dossiers = clients["numero_dossier"].to_numpy()
    revenu = _revenu_effectif(clients)
    candidats = np.array([i for i in range(len(clients)) if dossiers[i] not in exclus])
    ops, profils = [], []

    def tirer(part, masque=None):
        pool = candidats if masque is None else candidats[masque[candidats]]
        k = min(int(len(clients) * part), len(pool))
        choisis = rng.choice(pool, k, replace=False)
        return [i for i in choisis if dossiers[i] not in exclus]

    for i in tirer(PART_HERITAGE_REINVESTI):
        exclus.add(dossiers[i])
        entree = _date(rng, pd.Timestamp(DATE_DEBUT), 350)
        montant = rng.integers(200, 1500) * 100
        ops.append(_op(dossiers[i], entree, montant, "credit", "VIREMENT"))
        sortie = entree + pd.Timedelta(hours=float(rng.uniform(4, 46)))
        ops.append(_op(dossiers[i], sortie, montant * rng.uniform(0.92, 0.99), "debit", "VIREMENT"))
        profils.append({"numero_dossier": dossiers[i], "profil": "heritage_reinvesti"})

    for i in tirer(PART_FAMILLE_PAYS_RISQUE):
        exclus.add(dossiers[i])
        pays = ctx.pays(rng, "eleve")
        for mois in range(12):
            debut = pd.Timestamp(DATE_DEBUT) + pd.DateOffset(months=mois)
            ops.append(_op(dossiers[i], _date(rng, debut, 27), _lognormal_moyenne(rng, 250, 0.3),
                           "debit", "VIREMENT", pays))
        profils.append({"numero_dossier": dossiers[i], "profil": "famille_pays_a_risque"})

    for i in tirer(PART_GROS_INVESTISSEUR, masque=revenu >= 4_000):
        exclus.add(dossiers[i])
        for _ in range(rng.integers(6, 15)):
            mid, pays = ctx.marchand(rng, "crypto_change")
            ops.append(_op(dossiers[i], _date(rng, pd.Timestamp(DATE_DEBUT)),
                           _lognormal_moyenne(rng, 0.25 * revenu[i], 0.5), "debit", "ACHAT", pays, mid))
        profils.append({"numero_dossier": dossiers[i], "profil": "gros_investisseur_crypto"})

    return pd.DataFrame(ops, columns=COLONNES), pd.DataFrame(profils)


# ==========================================================================
# Injection des typologies
# ==========================================================================
# Chaque typologie existe en deux intensités :
# - "forte"    : le cas d'école, facile à repérer ;
# - "discrete" : moins d'opérations, des montants plus faibles, parfois juste
#                sous les seuils des règles. C'est là que le modèle doit apporter
#                quelque chose que les règles seules ne voient pas.
PART_DISCRETS = 0.5


def inject_fractionnement(rng, dossier, revenu, ctx, discret=False):
    """Dépôts d'espèces répétés juste sous le seuil d'alerte, puis sortie des fonds."""
    debut = _date(rng, pd.Timestamp(DATE_DEBUT), 290)
    ops = []
    nb = rng.integers(3, 6) if discret else rng.integers(8, 16)
    bas = 0.6 if discret else 0.8
    for _ in range(nb):
        montant = rng.integers(int(SEUIL_ALERTE_ESPECES * bas) // 10, SEUIL_ALERTE_ESPECES // 10) * 10
        ops.append(_op(dossier, _date(rng, debut, 90 if discret else 60), montant, "credit", "DEPOT_ESPECES"))
    if not discret or rng.random() < 0.5:
        total = sum(o["montant"] for o in ops)
        sortie = max(o["date_trans"] for o in ops) + pd.Timedelta(days=int(rng.integers(1, 6)))
        risque = str(rng.choice(["faible", "moyen", "eleve"], p=[.4, .4, .2]))
        ops.append(_op(dossier, sortie, total * rng.uniform(0.8, 0.95), "debit", "VIREMENT", ctx.pays(rng, risque)))
    return ops


def inject_compte_de_passage(rng, dossier, revenu, ctx, discret=False):
    """Entrée suivie d'une sortie quasi identique peu de temps après."""
    ops = []
    for _ in range(rng.integers(1, 3) if discret else rng.integers(3, 7)):
        entree = _date(rng, pd.Timestamp(DATE_DEBUT), 355)
        montant = rng.integers(30, 90) * 100 if discret else rng.integers(50, 400) * 100
        source = "FR" if rng.random() < 0.5 else ctx.pays(rng, str(rng.choice(["faible", "moyen"])))
        ops.append(_op(dossier, entree, montant, "credit", "VIREMENT", source))
        delai = rng.uniform(24, 96) if discret else rng.uniform(1, 47)
        sortie = entree + pd.Timedelta(hours=float(delai))
        destination = ctx.pays(rng, str(rng.choice(["faible", "moyen", "eleve"], p=[.3, .4, .3])))
        ops.append(_op(dossier, sortie, montant * rng.uniform(0.95, 0.995), "debit", "VIREMENT", destination))
    return ops


def inject_pays_a_risque(rng, dossier, revenu, ctx, discret=False):
    """Virements vers des pays à risque élevé, et retraits sur place."""
    ops = []
    nb, moyenne = (rng.integers(2, 4), 900) if discret else (rng.integers(4, 13), 2500)
    for _ in range(nb):
        date = _date(rng, pd.Timestamp(DATE_DEBUT))
        ops.append(_op(dossier, date, _lognormal_moyenne(rng, moyenne, 0.5), "debit",
                       "VIREMENT", ctx.pays(rng, "eleve")))
    for _ in range(rng.integers(0, 3) if discret else rng.integers(0, 6)):
        date = _date(rng, pd.Timestamp(DATE_DEBUT))
        ops.append(_op(dossier, date, rng.integers(20, 51) * 10, "debit", "DAB", ctx.pays(rng, "eleve")))
    return ops


def inject_incoherence_revenus(rng, dossier, revenu, ctx, discret=False):
    """Entrées d'argent sans rapport avec le revenu déclaré, dépensées en biens de valeur."""
    ops = []
    debut_mois = int(rng.integers(0, 6))
    nb_mois = rng.integers(2, 5) if discret else rng.integers(4, 11)
    facteur = 0.8 if discret else 1.5
    for mois in range(debut_mois, debut_mois + int(nb_mois)):
        debut = pd.Timestamp(DATE_DEBUT) + pd.DateOffset(months=min(mois, 11))
        for _ in range(rng.integers(1, 3) if discret else rng.integers(2, 5)):
            ops.append(_op(dossier, _date(rng, debut, 27), _lognormal_moyenne(rng, facteur * revenu, 0.4),
                           "credit", "VIREMENT"))
    for _ in range(rng.integers(1, 3) if discret else rng.integers(3, 9)):
        mid, pays = ctx.marchand(rng, "valeurs")
        ops.append(_op(dossier, _date(rng, pd.Timestamp(DATE_DEBUT)),
                       _lognormal_moyenne(rng, 1200 if discret else 2500, 0.5), "debit", "ACHAT", pays, mid))
    return ops


def inject_crypto(rng, dossier, revenu, ctx, discret=False):
    """Achats crypto répétés au regard du revenu, souvent alimentés par des espèces."""
    ops = []
    nb, moyenne = (rng.integers(3, 7), 1000) if discret else (rng.integers(8, 26), 2500)
    for _ in range(nb):
        date = _date(rng, pd.Timestamp(DATE_DEBUT) + pd.Timedelta(days=3), 360)
        montant = min(_lognormal_moyenne(rng, moyenne, 0.6), 15_000)
        if rng.random() < 0.5:
            depot = date - pd.Timedelta(days=int(rng.integers(1, 4)))
            ops.append(_op(dossier, depot, max(round(montant * rng.uniform(0.8, 1.0), -1), 10),
                           "credit", "DEPOT_ESPECES"))
        mid, pays = ctx.marchand(rng, "crypto_change")
        ops.append(_op(dossier, date, montant, "debit", "ACHAT", pays, mid))
    if not discret and rng.random() < 0.3:
        for _ in range(rng.integers(1, 4)):
            ops.append(_op(dossier, _date(rng, pd.Timestamp(DATE_DEBUT)), _lognormal_moyenne(rng, 3000, 0.5),
                           "debit", "VIREMENT", ctx.pays(rng, "eleve")))
    return ops


INJECTEURS = {
    "fractionnement": inject_fractionnement,
    "compte_de_passage": inject_compte_de_passage,
    "pays_a_risque": inject_pays_a_risque,
    "incoherence_revenus": inject_incoherence_revenus,
    "crypto": inject_crypto,
}


def _choisir_clients_suspects(rng, clients: pd.DataFrame) -> dict:
    """150 clients distincts. L'incohérence de revenus vise des revenus modestes."""
    revenu = _revenu_effectif(clients)
    modestes = clients.index[(revenu < 1_800) & (clients["ccsp"] != "Retraités")].to_numpy()
    choix = {"incoherence_revenus": rng.choice(modestes, N_PAR_TYPOLOGIE, replace=False)}
    restants = np.setdiff1d(clients.index.to_numpy(), choix["incoherence_revenus"])
    tires = rng.choice(restants, N_PAR_TYPOLOGIE * 4, replace=False).reshape(4, -1)
    autres = [t for t in TYPOLOGIES if t != "incoherence_revenus"]
    choix.update(dict(zip(autres, tires)))
    return choix


# ==========================================================================
# Assemblage
# ==========================================================================
def build_transactions(clients, marchands, mcc, pays):
    """Renvoie (transactions, labels_clients, labels_transactions, profils_legitimes)."""
    rng = get_rng(offset=3)
    ctx = _Contexte(marchands, mcc, pays)

    normales = _activite_normale(rng, clients, ctx)

    revenu = _revenu_effectif(clients)
    injectees, labels_clients = [], []
    for typologie, indices in _choisir_clients_suspects(rng, clients).items():
        for k, i in enumerate(indices):
            discret = k < len(indices) * PART_DISCRETS
            dossier = clients.at[i, "numero_dossier"]
            ops = INJECTEURS[typologie](rng, dossier, revenu[i], ctx, discret=discret)
            injectees.extend({**op, "typologie": typologie} for op in ops)
            labels_clients.append({"numero_dossier": dossier, "typologie": typologie,
                                   "intensite": "discrete" if discret else "forte"})
    injectees = pd.DataFrame(injectees)
    labels_clients = pd.DataFrame(labels_clients)

    faux_amis, profils_legitimes = _faux_amis(rng, clients, ctx, set(labels_clients["numero_dossier"]))

    tout = pd.concat([normales.assign(typologie=None), faux_amis.assign(typologie=None), injectees],
                     ignore_index=True)
    tout["date_trans"] = pd.to_datetime(tout["date_trans"]).dt.floor("s")
    tout = tout.sort_values(["date_trans", "numero_dossier"], kind="stable").reset_index(drop=True)
    tout["transaction_id"] = [f"T{i + 1:08d}" for i in range(len(tout))]

    transactions = tout[["transaction_id"] + COLONNES]
    labels_transactions = tout.loc[tout["typologie"].notna(), ["transaction_id", "numero_dossier", "typologie"]]

    assert transactions["transaction_id"].is_unique
    assert (transactions["montant"] > 0).all(), "Montant nul ou négatif"
    assert labels_clients["numero_dossier"].is_unique, "Un client a deux typologies"
    assert not set(profils_legitimes["numero_dossier"]) & set(labels_clients["numero_dossier"])
    return transactions, labels_clients, labels_transactions.reset_index(drop=True), profils_legitimes


if __name__ == "__main__":
    from src.generate.config import RAW_DIR, save

    clients = pd.read_csv(RAW_DIR / "clients.csv")
    marchands = pd.read_csv(RAW_DIR / "ref_marchands.csv")
    mcc = pd.read_csv(RAW_DIR / "ref_mcc.csv")
    pays = pd.read_csv(RAW_DIR / "ref_pays.csv")

    transactions, labels_clients, labels_transactions, profils = build_transactions(clients, marchands, mcc, pays)
    save(transactions, "transactions")
    save(labels_clients, "labels_clients")
    save(labels_transactions, "labels_transactions")
    save(profils, "profils_legitimes")

    print(transactions.groupby(["type_trans", "sens"])["montant"].agg(["count", "median"]).round(0).to_string())
    print(f"\n{len(transactions):,} transactions, {len(labels_clients)} clients suspects")
    print(labels_transactions["typologie"].value_counts().to_string())
