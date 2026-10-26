"""Paramètres communs à tous les générateurs de données synthétiques."""

from pathlib import Path

import numpy as np

# Graine unique : mêmes paramètres => mêmes données, à chaque exécution.
SEED = 42

N_CLIENTS = 5_000
N_MARCHANDS = 200

# Période couverte par les transactions (utilisée à partir de l'étape transactions).
DATE_DEBUT = "2025-01-01"
DATE_FIN = "2025-12-31"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_rng(offset: int = 0) -> np.random.Generator:
    """Générateur aléatoire reproductible.

    Chaque table utilise son propre offset pour que modifier une table
    ne change pas les tirages des autres.
    """
    return np.random.default_rng(SEED + offset)


def save(df, name: str) -> Path:
    """Sauvegarde une table en CSV dans data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
