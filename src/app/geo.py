"""Cartes des flux par pays (Plotly, sans clé d'API ni fond de carte externe).

Les pays sont placés sur leur capitale : les petits territoires (Monaco, îles
Caïmans) n'existent pas toujours dans les fonds de carte par pays, des
coordonnées explicites évitent qu'ils disparaissent.
"""

import pandas as pd
import plotly.graph_objects as go

from src.app import theme

ORIGINE = ("FR", 48.86, 2.35)

COORDONNEES = {
    "FR": (48.86, 2.35), "BE": (50.85, 4.35), "DE": (52.52, 13.40), "ES": (40.42, -3.70),
    "IT": (41.90, 12.50), "PT": (38.72, -9.14), "NL": (52.37, 4.90), "CH": (46.95, 7.45),
    "GB": (51.51, -0.13), "US": (38.90, -77.04), "CA": (45.42, -75.70), "MA": (34.02, -6.84),
    "TN": (36.81, 10.18), "SN": (14.72, -17.47), "TR": (39.93, 32.86), "CN": (39.90, 116.40),
    "MC": (43.74, 7.42), "AE": (24.45, 54.38), "PA": (8.98, -79.52), "KY": (19.29, -81.37),
    "CI": (5.36, -4.01), "DZ": (36.75, 3.06), "LB": (33.89, 35.50), "VE": (10.48, -66.90),
    "YE": (15.37, 44.19), "SY": (33.51, 36.29), "HT": (18.59, -72.31), "KP": (39.04, 125.76),
    "IR": (35.69, 51.39), "MM": (19.76, 96.08),
}

COULEURS_RISQUE = theme.RISQUE
NOMS_RISQUE = {"eleve": "Risque élevé", "moyen": "Risque moyen", "faible": "Risque faible"}


def _taille(montants: pd.Series, mini=8, maxi=40) -> pd.Series:
    """Taille de bulle proportionnelle à la racine du montant (lecture des surfaces)."""
    racine = montants.clip(lower=0) ** 0.5
    if racine.max() == racine.min():
        return pd.Series(mini + (maxi - mini) / 2, index=montants.index)
    return mini + (racine - racine.min()) / (racine.max() - racine.min()) * (maxi - mini)


def _fond(fig: go.Figure, hauteur: int) -> go.Figure:
    fig.update_geos(
        projection_type="natural earth", showcountries=True, countrycolor=theme.BORDURE,
        showland=True, landcolor=theme.SURFACE_HAUTE, showocean=True, oceancolor=theme.FOND,
        showlakes=False, showframe=False, coastlinecolor=theme.BORDURE, bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=hauteur, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=-0.02, x=0))
    return theme.styliser(fig)


def _bulles(fig, agg: pd.DataFrame, survol: str):
    for risque in ["eleve", "moyen", "faible"]:
        groupe = agg[agg["risque_pays"] == risque]
        if groupe.empty:
            continue
        lat, lon = zip(*groupe["code_pays"].map(COORDONNEES))
        fig.add_trace(go.Scattergeo(
            lat=lat, lon=lon, mode="markers", name=NOMS_RISQUE.get(risque, risque),
            marker=dict(size=_taille(agg["montant"]).loc[groupe.index], color=COULEURS_RISQUE.get(risque, "#8A94A6"),
                        opacity=0.8, line=dict(width=1, color=theme.FOND)),
            customdata=groupe[["pays", "montant", "nb_operations"] + (["nb_clients"] if "nb_clients" in groupe else [])],
            hovertemplate=survol,
        ))


def carte_client(tx: pd.DataFrame) -> tuple:
    """Carte des opérations hors de France d'un client, avec les flux depuis la France.

    Renvoie (figure ou None, tableau par pays).
    """
    agg = (tx.groupby(["code_pays", "pays", "risque_pays"], dropna=False)
             .agg(montant=("montant", "sum"), nb_operations=("transaction_id", "count"),
                  entrees=("montant", lambda m: m[tx.loc[m.index, "sens"] == "credit"].sum()),
                  sorties=("montant", lambda m: m[tx.loc[m.index, "sens"] == "debit"].sum()))
             .reset_index().sort_values("montant", ascending=False))
    etranger = agg[(agg["code_pays"] != "FR") & agg["code_pays"].isin(COORDONNEES)].reset_index(drop=True)
    if etranger.empty:
        return None, agg

    fig = go.Figure()
    for _, ligne in etranger.iterrows():
        lat, lon = COORDONNEES[ligne["code_pays"]]
        fig.add_trace(go.Scattergeo(
            lat=[ORIGINE[1], lat], lon=[ORIGINE[2], lon], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(width=1.5, color=COULEURS_RISQUE.get(ligne["risque_pays"], "#8A94A6")), opacity=0.5,
        ))
    fig.add_trace(go.Scattergeo(
        lat=[ORIGINE[1]], lon=[ORIGINE[2]], mode="markers", name="France (compte)",
        marker=dict(size=10, color=theme.TEXTE, symbol="diamond"), hoverinfo="name",
    ))
    _bulles(fig, etranger, "<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} €<br>"
                           "%{customdata[2]} opérations<extra></extra>")
    return _fond(fig, 380), agg


def carte_alertes(flux: pd.DataFrame):
    """Vue d'ensemble : flux hors de France des clients en alerte, agrégés par pays."""
    flux = flux[flux["code_pays"].isin(COORDONNEES)].reset_index(drop=True)
    if flux.empty:
        return None
    fig = go.Figure()
    _bulles(fig, flux, "<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} €<br>"
                       "%{customdata[2]} opérations · %{customdata[3]} clients<extra></extra>")
    return _fond(fig, 360)
