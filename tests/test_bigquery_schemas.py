"""Vérifie que les schémas BigQuery suivent les colonnes des CSV.

BigQuery charge les CSV par position : une colonne ajoutée dans un générateur
sans mise à jour du schéma fait échouer le chargement. Ce test le détecte en local.
"""

import pandas as pd
import pytest

from src.bigquery.load_raw import TABLES
from src.generate.config import RAW_DIR


@pytest.mark.parametrize("table", list(TABLES))
def test_schema_identique_aux_colonnes_du_csv(table):
    chemin = RAW_DIR / f"{table}.csv"
    if not chemin.exists():
        pytest.skip("Lancer d'abord python -m src.generate.run_all")
    _, schema = TABLES[table]
    colonnes_csv = list(pd.read_csv(chemin, nrows=1).columns)
    assert colonnes_csv == [champ.name for champ in schema]
