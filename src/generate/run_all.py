"""Génère toutes les tables brutes dans l'ordre des dépendances."""

from src.generate.clients import build_clients
from src.generate.config import save
from src.generate.ref_marchands import build_ref_marchands
from src.generate.ref_mcc import build_ref_mcc
from src.generate.ref_pays import build_ref_pays
from src.generate.transactions import build_transactions


def _sauver(nom, df):
    path = save(df, nom)
    print(f"{nom:<20} {len(df):>9,} lignes -> {path}")


if __name__ == "__main__":
    mcc = build_ref_mcc()
    pays = build_ref_pays()
    marchands = build_ref_marchands()
    clients = build_clients()
    for nom, df in [("ref_mcc", mcc), ("ref_pays", pays), ("ref_marchands", marchands), ("clients", clients)]:
        _sauver(nom, df)

    transactions, labels_clients, labels_transactions, profils = build_transactions(clients, marchands, mcc, pays)
    _sauver("transactions", transactions)
    _sauver("labels_clients", labels_clients)
    _sauver("labels_transactions", labels_transactions)
    _sauver("profils_legitimes", profils)
