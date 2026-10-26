"""Thème sombre de l'interface : palette, CSS et style des graphiques Plotly.

Toutes les couleurs sont définies ici, pour garder une interface cohérente.
"""

import plotly.graph_objects as go

# Palette
FOND = "#0B1220"            # fond principal (bleu nuit)
SURFACE = "#121B2D"         # cartes, barre latérale
SURFACE_HAUTE = "#18233A"   # survols, en-têtes
BORDURE = "#24314A"
TEXTE = "#E6EAF2"
TEXTE_SECONDAIRE = "#93A0B8"
GRILLE = "#1E2A40"

ACCENT = "#4C8DFF"          # bleu : entrées, éléments actifs
DANGER = "#F2545B"          # rouge : sorties, risque élevé
ALERTE = "#F5A524"          # orange : risque moyen
NEUTRE = "#7C8799"          # gris : contributions négatives, références
SUCCES = "#3CCB7F"

RISQUE = {"eleve": DANGER, "moyen": ALERTE, "faible": ACCENT}
METHODES = [NEUTRE, ALERTE, ACCENT]  # Isolation Forest, Règles, Hybride

CSS = f"""
<style>
  .block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}}
  h1 {{font-size: 1.9rem !important; font-weight: 700 !important; letter-spacing: -0.02em;}}
  h4 {{font-size: 1.05rem !important; font-weight: 600 !important; color: {TEXTE};
       margin-top: 1.4rem !important; padding-bottom: 0.3rem; border-bottom: 1px solid {BORDURE};}}

  /* Indicateurs en cartes */
  div[data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid {BORDURE}; border-radius: 10px;
       padding: 0.8rem 1rem;}}
  div[data-testid="stMetricLabel"] p {{color: {TEXTE_SECONDAIRE}; font-size: 0.8rem; text-transform: uppercase;
       letter-spacing: 0.04em;}}
  div[data-testid="stMetricValue"] {{font-size: 1.5rem; font-weight: 600;}}

  /* Blocs */
  div[data-testid="stExpander"] details {{background: {SURFACE}; border: 1px solid {BORDURE}; border-radius: 10px;}}
  div[data-testid="stPlotlyChart"] {{background: {SURFACE}; border: 1px solid {BORDURE}; border-radius: 10px;
       padding: 0.4rem;}}
  section[data-testid="stSidebar"] {{border-right: 1px solid {BORDURE};}}

  /* Badges de règles */
  .badge {{display:inline-block; padding:3px 12px; margin:3px 6px 3px 0; border-radius:999px;
           background: rgba(242,84,91,0.12); color: #FF8A8F; border: 1px solid rgba(242,84,91,0.35);
           font-size: 0.82rem;}}
  .badge-ok {{background: rgba(60,203,127,0.12); color: {SUCCES}; border-color: rgba(60,203,127,0.35);}}

  /* Encadré de la note IA */
  .carte {{border: 1px dashed {BORDURE}; border-radius: 10px; padding: 0.9rem 1.1rem; background: {SURFACE};
           color: {TEXTE_SECONDAIRE};}}

  /* En-têtes de page */
  .entete {{margin: 0.2rem 0 1.2rem 0;}}
  .sur-titre {{color: {ACCENT}; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
               text-transform: uppercase; margin-bottom: 0.25rem;}}
  .titre-page {{color: {TEXTE}; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2;
                display: flex; align-items: center; gap: 0.8rem;}}
  .sous-titre {{color: {TEXTE_SECONDAIRE}; font-size: 0.92rem; margin-top: 0.3rem;}}

  /* Puces de statut */
  .puce {{font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em; padding: 3px 10px; border-radius: 999px;
          border: 1px solid {BORDURE}; color: {TEXTE_SECONDAIRE}; background: {SURFACE};}}
  .puce-attente {{color: #FFC861; border-color: rgba(245,165,36,0.4); background: rgba(245,165,36,0.10);}}
  .puce-cours {{color: #8DB6FF; border-color: rgba(76,141,255,0.4); background: rgba(76,141,255,0.10);}}
  .puce-ok {{color: {SUCCES}; border-color: rgba(60,203,127,0.4); background: rgba(60,203,127,0.10);}}
  .puce-alerte {{color: #FF8A8F; border-color: rgba(242,84,91,0.4); background: rgba(242,84,91,0.10);}}

  /* Éléments Streamlit qui font "prototype" */
  [data-testid="stHeaderActionElements"] {{display: none;}}
  [data-testid="stDecoration"] {{display: none;}}
  footer {{visibility: hidden;}}
  [data-testid="stSidebarNavSeparator"] {{border-color: {BORDURE};}}
  [data-testid="stNavSectionHeader"] {{color: {TEXTE_SECONDAIRE}; font-size: 0.72rem; letter-spacing: 0.08em;
       text-transform: uppercase;}}
</style>
"""


def styliser(fig: go.Figure) -> go.Figure:
    """Fond transparent, texte clair et grille discrète, pour s'intégrer au thème."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXTE_SECONDAIRE, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTE_SECONDAIRE)),
        hoverlabel=dict(bgcolor=SURFACE_HAUTE, bordercolor=BORDURE, font=dict(color=TEXTE)),
        title_font=dict(color=TEXTE, size=14),
    )
    fig.update_xaxes(gridcolor=GRILLE, zeroline=False, linecolor=BORDURE)
    fig.update_yaxes(gridcolor=GRILLE, zeroline=False, linecolor=BORDURE)
    return fig
