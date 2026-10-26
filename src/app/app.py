"""Interface de l'analyste conformité.

Lancement, depuis la racine du projet :
    streamlit run src/app/app.py

Trois pages :
- File d'alertes : les clients triés par score, avec filtres ;
- Fiche client   : scores, règles déclenchées, explications SHAP, flux, transactions,
                   note d'analyse et décision de l'analyste ;
- Performance    : évaluation contre la vérité terrain (mode démo).
"""

import hmac
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.app.donnees import (  # noqa: E402
    LIBELLES_REGLES, NOMS_COURTS, charger_scores, flux_par_pays, charger_shap, charger_verite_terrain,
    libelle_variable, transactions_client,
)
from src.agent.graphe import MODELE_PAR_DEFAUT, generer_note  # noqa: E402
from src.app import journal, theme  # noqa: E402
from src.app.geo import NOMS_RISQUE, carte_alertes, carte_client  # noqa: E402
from src.detection.evaluate import SCORES  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="Analyste LCB-FT", page_icon=":material/policy:", layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icone.svg"), size="large")

BLEU, CORAIL, GRIS = theme.ACCENT, theme.DANGER, theme.NEUTRE
TYPES = {"ACHAT": "Achat carte", "DAB": "Retrait DAB", "VIREMENT": "Virement", "DEPOT_ESPECES": "Dépôt espèces"}
NOMS_TYPOLOGIES = {
    "fractionnement": "Fractionnement", "compte_de_passage": "Compte de passage",
    "pays_a_risque": "Pays à risque", "incoherence_revenus": "Incohérence de revenus", "crypto": "Crypto-actifs",
}
NOMS_PROFILS = {
    "heritage_reinvesti": "Héritage vite réinvesti", "famille_pays_a_risque": "Envois à la famille (pays à risque)",
    "gros_investisseur_crypto": "Gros investisseur crypto (hauts revenus)",
}
DECISIONS = ["À traiter", "En cours d'analyse", "Classé sans suite", "Escaladé (déclaration de soupçon)"]

st.markdown(theme.CSS, unsafe_allow_html=True)

def statut(dossier: str) -> str:
    return journal.statuts().get(dossier, DECISIONS[0])


CLASSES_STATUT = {
    "À traiter": "puce-attente", "En cours d'analyse": "puce-cours",
    "Classé sans suite": "puce-ok", "Escaladé (déclaration de soupçon)": "puce-alerte",
}


def entete(sur_titre: str, titre: str, sous_titre: str = "", puce: str = "") -> None:
    """En-tête de page : sur-titre, titre, sous-titre et puce de statut optionnelle."""
    puce_html = f'<span class="puce {CLASSES_STATUT.get(puce, "")}">{puce}</span>' if puce else ""
    st.markdown(
        f'<div class="entete"><div class="sur-titre">{sur_titre}</div>'
        f'<div class="titre-page">{titre}{puce_html}</div>'
        f'<div class="sous-titre">{sous_titre}</div></div>',
        unsafe_allow_html=True,
    )


def badges(codes, classe="badge") -> str:
    if not codes:
        return '<span class="badge badge-ok">Aucune règle déclenchée</span>'
    return "".join(f'<span class="{classe}">{LIBELLES_REGLES.get(c, c)}</span>' for c in codes)


# ==========================================================================
# Page 1 : file d'alertes
# ==========================================================================
def page_alertes():
    scores = charger_scores()
    entete("Surveillance", "File d'alertes",
           "Clients triés par score de risque : 50 % Isolation Forest, 50 % règles métier.")

    with st.sidebar:
        st.subheader("Filtres")
        seuil = st.slider("Score minimum", 0, 100, 60)
        regles = st.multiselect("Règle déclenchée", list(LIBELLES_REGLES), placeholder="Toutes les règles",
                                format_func=lambda c: NOMS_COURTS[c])
        csp = st.multiselect("CSP", sorted(scores["ccsp"].dropna().unique()), placeholder="Toutes les CSP")
        statuts = st.multiselect("Statut", DECISIONS, placeholder="Tous les statuts")

    vue = scores[scores["score_final"] >= seuil]
    if regles:
        vue = vue[vue["regles_declenchees"].apply(lambda r: any(c in r for c in regles))]
    if csp:
        vue = vue[vue["ccsp"].isin(csp)]
    vue = vue.assign(statut=vue["numero_dossier"].map(statut))
    if statuts:
        vue = vue[vue["statut"].isin(statuts)]

    traites = sum(d in DECISIONS[2:] for d in journal.statuts().values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients surveillés", f"{len(scores):,}".replace(",", " "))
    c2.metric("Alertes affichées", len(vue))
    c3.metric("Score moyen des alertes", f"{vue['score_final'].mean():.0f}" if len(vue) else "-")
    c4.metric("Dossiers tranchés", traites)

    with st.expander("Vue géographique : où partent les fonds des clients en alerte", expanded=True):
        flux = flux_par_pays(tuple(vue["numero_dossier"]))
        if flux.empty:
            st.info("Aucun flux hors de France pour les alertes affichées.")
        else:
            gauche, droite = st.columns([3, 2])
            with gauche:
                st.plotly_chart(carte_alertes(flux), width="stretch")
            with droite:
                risque = flux[flux["risque_pays"] == "eleve"]
                st.metric("Montant vers des pays à risque élevé", f"{risque['montant'].sum():,.0f} €".replace(",", " "),
                          help=f"{risque['nb_clients'].sum()} relations client-pays concernées")
                st.dataframe(
                    flux.head(10).assign(risque_pays=flux["risque_pays"].map(NOMS_RISQUE))[
                        ["pays", "risque_pays", "montant", "nb_clients", "nb_operations"]
                    ].rename(columns={"pays": "Pays", "risque_pays": "Risque", "montant": "Montant (€)",
                                      "nb_clients": "Clients", "nb_operations": "Opérations"}),
                    hide_index=True, width="stretch", height=280,
                    column_config={"Montant (€)": st.column_config.NumberColumn(format="%.0f")},
                )
            st.caption("Opérations hors de France des clients affichés dans la file (filtres appliqués). "
                       "Taille des bulles : montant. Couleur : niveau de risque du pays.")

    tableau = pd.DataFrame({
        "Rang": vue["rang"],
        "Dossier": vue["numero_dossier"],
        "Client": vue["prenom"] + " " + vue["nom"],
        "CSP": vue["ccsp"],
        "Score": vue["score_final"],
        "Modèle": vue["score_if"],
        "Règles": vue["score_regles"],
        "Règles déclenchées": vue["regles_declenchees"].apply(lambda r: ", ".join(NOMS_COURTS[c] for c in r)),
        "Statut": vue["statut"],
    })

    if mode_demo():
        labels, _, profils = charger_verite_terrain()
        verite = vue["numero_dossier"].map(labels.set_index("numero_dossier")["typologie"])
        legitime = vue["numero_dossier"].map(profils.set_index("numero_dossier")["profil"])
        tableau["Vérité terrain"] = verite.fillna(legitime.map(lambda p: f"légitime : {p}" if isinstance(p, str) else "normal"))

    st.write("Cliquez sur une ligne pour ouvrir la fiche client.")
    selection = st.dataframe(
        tableau, hide_index=True, width="stretch", height=520,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Modèle": st.column_config.NumberColumn(format="%.0f"),
            "Règles": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    lignes = selection.selection.rows
    if lignes:
        st.session_state.dossier = tableau.iloc[lignes[0]]["Dossier"]
        st.switch_page(PAGE_FICHE)


# ==========================================================================
# Page 2 : fiche client
# ==========================================================================
def graphique_shap(dossier: str):
    contributions = charger_shap().loc[dossier]
    contributions.index = [libelle_variable(c.removeprefix("shap_")) for c in contributions.index]
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(10)[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=[CORAIL if v > 0 else GRIS for v in top.values],
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=20, t=20, b=50), xaxis_title="Contribution SHAP")
    theme.styliser(fig)
    return fig


def graphique_flux(tx: pd.DataFrame, revenu_mensuel: float):
    mensuel = (tx.assign(mois=tx["date_trans"].dt.to_period("M").dt.to_timestamp())
                 .groupby(["mois", "sens"])["montant"].sum().unstack(fill_value=0)
                 .reindex(columns=["credit", "debit"], fill_value=0))
    fig = go.Figure()
    fig.add_bar(x=mensuel.index, y=mensuel["credit"], name="Entrées", marker_color=BLEU)
    fig.add_bar(x=mensuel.index, y=mensuel["debit"], name="Sorties", marker_color=CORAIL)
    if revenu_mensuel > 0:
        fig.add_hline(y=revenu_mensuel, line_dash="dash", line_color=GRIS,
                      annotation_text="Revenu déclaré", annotation_position="top right",
                      annotation_font_color=GRIS)
    fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=20, t=40, b=50),
                      legend=dict(orientation="h", y=1.1), yaxis_title="€",
                      xaxis=dict(tickformat="%m/%Y", dtick="M2"))
    theme.styliser(fig)
    return fig


def page_fiche():
    scores = charger_scores()
    dossiers = scores["numero_dossier"].tolist()
    defaut = st.session_state.get("dossier", dossiers[0])
    dossier = st.sidebar.selectbox("Dossier", dossiers, index=dossiers.index(defaut),
                                   format_func=lambda d: f"{d} (rang {int(scores.set_index('numero_dossier').at[d, 'rang'])})")
    st.session_state.dossier = dossier
    client = scores.set_index("numero_dossier").loc[dossier]
    tx = transactions_client(dossier)

    entete(f"Fiche client · {dossier}", f"{client['prenom']} {client['nom']}",
           f"{client['ccsp']} · {int(client['age'])} ans · client depuis {client['date_ouverture'][:4]}",
           puce=statut(dossier))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score de risque", f"{client['score_final']:.0f} / 100")
    c2.metric("Rang", f"{int(client['rang'])} / {len(scores)}")
    c3.metric("Score modèle", f"{client['score_if']:.0f}")
    c4.metric("Score règles", f"{client['score_regles']:.0f}")
    revenu = float(client["salaire_declare"] + client["pension"])
    c5.metric("Revenu mensuel", f"{revenu:,.0f} €".replace(",", " "))

    st.markdown("#### Pourquoi ce client est en alerte")
    st.markdown(badges(client["regles_declenchees"]), unsafe_allow_html=True)

    gauche, droite = st.columns([1, 1])
    with gauche:
        st.markdown("**Ce qui rend le profil atypique (modèle)**")
        st.plotly_chart(graphique_shap(dossier), width="stretch")
        st.caption("En rouge : ce qui augmente le risque. En gris : ce qui le diminue.")
    with droite:
        st.markdown("**Entrées et sorties par mois**")
        st.plotly_chart(graphique_flux(tx, revenu), width="stretch")

    st.markdown("#### Où le client a effectué ses opérations")
    figure, par_pays = carte_client(tx)
    if figure is None:
        st.info("Toutes les opérations de ce client ont lieu en France.")
    else:
        gauche, droite = st.columns([3, 2])
        with gauche:
            st.plotly_chart(figure, width="stretch")
        with droite:
            st.dataframe(
                par_pays.assign(risque_pays=par_pays["risque_pays"].map(NOMS_RISQUE))[
                    ["pays", "risque_pays", "nb_operations", "entrees", "sorties"]
                ].rename(columns={"pays": "Pays", "risque_pays": "Risque", "nb_operations": "Opérations",
                                  "entrees": "Entrées (€)", "sorties": "Sorties (€)"}),
                hide_index=True, width="stretch", height=340,
                column_config={"Entrées (€)": st.column_config.NumberColumn(format="%.0f"),
                               "Sorties (€)": st.column_config.NumberColumn(format="%.0f")},
            )

    st.markdown("#### Transactions")
    filtre = st.radio("Afficher", ["Opérations signalées", "Toutes"], horizontal=True, label_visibility="collapsed")
    vue = tx[tx["signal"] != ""] if filtre == "Opérations signalées" else tx
    colonnes = ["date_trans", "type_trans", "sens", "montant", "marchand", "activite", "pays", "signal"]
    if mode_demo():
        _, labels_tx, _ = charger_verite_terrain()
        vue = vue.merge(labels_tx[["transaction_id", "typologie"]], on="transaction_id", how="left")
        colonnes.append("typologie")
    vue = vue.assign(marchand=vue["marchand"].fillna("—"), activite=vue["activite"].fillna("—"),
                     sens=vue["sens"].map({"credit": "Entrée", "debit": "Sortie"}),
                     type_trans=vue["type_trans"].map(TYPES).fillna(vue["type_trans"]))
    st.dataframe(
        vue[colonnes].rename(columns={"date_trans": "Date", "type_trans": "Type", "sens": "Sens",
                                      "montant": "Montant (€)", "marchand": "Marchand", "activite": "Activité",
                                      "pays": "Pays", "signal": "Signal", "typologie": "Vérité terrain"}),
        hide_index=True, width="stretch", height=300,
        column_config={"Montant (€)": st.column_config.NumberColumn(format="%.2f"),
                       "Date": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm")},
    )
    st.caption(f"{len(vue)} opérations affichées sur {len(tx)}.")

    st.markdown("#### Note d'analyse")
    section_note(dossier, client, tx)

    st.markdown("#### Décision de l'analyste")
    with st.form(f"decision_{dossier}", clear_on_submit=True):
        decision = st.radio("Décision", DECISIONS, index=DECISIONS.index(statut(dossier)), horizontal=True)
        justification = st.text_area("Justification",
                                     placeholder="Ex. : justificatif de succession reçu, flux cohérent avec la vente.")
        if st.form_submit_button("Enregistrer la décision", type="primary", icon=":material/save:"):
            if decision in DECISIONS[2:] and len(justification.strip()) < 10:
                st.error("Un classement ou une escalade doit être motivé : ajoutez une justification.")
            else:
                journal.enregistrer(dossier, decision, justification, st.session_state.get("analyste", ""),
                                    client["score_final"], dossier in st.session_state.get("notes", {}))
                st.success("Décision enregistrée dans le journal.")
                st.rerun()

    historique = journal.historique(dossier)
    st.markdown("#### Historique du dossier")
    if historique.empty:
        st.caption("Aucune décision enregistrée pour ce dossier.")
    else:
        st.dataframe(
            historique.assign(horodatage=historique["horodatage"].dt.tz_convert("Europe/Paris").dt.tz_localize(None))[
                ["horodatage", "decision", "justification", "analyste", "score_final", "note_generee"]
            ].rename(columns={"horodatage": "Date", "decision": "Décision", "justification": "Justification",
                              "analyste": "Analyste", "score_final": "Score", "note_generee": "Note IA"}),
            hide_index=True, width="stretch",
            column_config={"Date": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                           "Score": st.column_config.NumberColumn(format="%.0f")},
        )


def note_pour_affichage(note: str, documents: list) -> str:
    """Mise en forme de lecture : les identifiants de sources [reg_...] deviennent des renvois
    numérotés, et la section Références une liste numérotée (titre et texte de référence).
    La note brute, avec ses identifiants, reste celle qui est contrôlée et journalisée."""
    par_id = {d.id: d for d in documents}
    corps = re.split(r"^##\s+Références\s*$", note, maxsplit=1, flags=re.MULTILINE)[0]
    ordre = []
    for ident in re.findall(r"\[([a-z_]+)\]", corps):
        if ident in par_id and ident not in ordre:
            ordre.append(ident)

    def renvoi(m):
        ident = m.group(1)
        if ident not in ordre:
            return m.group(0)
        n = ordre.index(ident) + 1
        return f'<sup title="{par_id[ident].titre}" style="color:{theme.ACCENT}; font-weight:600;">{n}</sup>'

    corps = re.sub(r"\s?\[([a-z_]+)\]", renvoi, corps)
    references = "\n".join(f"{i}. **{par_id[ident].titre}** — {par_id[ident].source}"
                            for i, ident in enumerate(ordre, start=1))
    texte = corps.rstrip() + ("\n\n## Références\n" + references if references else "")
    # Titres de la note un cran plus petits que ceux de la page.
    return re.sub(r"^##\s+", "##### ", texte, flags=re.MULTILINE)


def section_note(dossier: str, client: pd.Series, tx: pd.DataFrame) -> None:
    """Note rédigée par l'agent (LangGraph + RAG + Gemini), avec ses sources et ses contrôles."""
    notes = st.session_state.setdefault("notes", {})
    simule = os.environ.get("AGENT_LLM", "vertex") == "simule"
    modele = "rédacteur simulé (hors ligne)" if simule else os.environ.get("GEMINI_MODEL", MODELE_PAR_DEFAUT)

    gauche, droite = st.columns([3, 1])
    with gauche:
        st.caption(f"Rédigée par l'agent à partir des faits du dossier et de la base réglementaire · modèle : {modele}. "
                   "La note aide l'analyste ; la décision lui appartient.")
    with droite:
        libelle = "Régénérer la note" if dossier in notes else "Générer la note d'analyse"
        lancer = st.button(libelle, icon=":material/auto_awesome:", type="primary", width="stretch")

    if lancer:
        with st.spinner("Collecte des faits, recherche réglementaire, rédaction et contrôles..."):
            try:
                donnees_client = client.to_dict() | {"numero_dossier": dossier}
                notes[dossier] = generer_note(donnees_client, tx)
            except Exception as erreur:  # noqa: BLE001 — on affiche l'erreur à l'analyste
                st.error(f"La génération a échoué : {erreur}")
                st.caption("Vérifiez `gcloud auth application-default login` et l'API Vertex AI, "
                           "ou lancez l'application avec `AGENT_LLM=simule RAG_MOTEUR=local` pour le mode hors ligne.")
                return

    resultat = notes.get(dossier)
    if resultat is None:
        st.markdown('<div class="carte">Cliquez sur « Générer la note d\'analyse » : l\'agent rassemble les faits '
                    'chiffrés, retrouve les textes et typologies applicables, rédige la note puis la contrôle '
                    '(sources, montants, articles cités, confidentialité).</div>', unsafe_allow_html=True)
        return

    with st.container(border=True):
        st.markdown(note_pour_affichage(resultat["note"], resultat["documents"]), unsafe_allow_html=True)
    if resultat["problemes"]:
        st.warning("Points à vérifier signalés par les contrôles automatiques :\n- "
                   + "\n- ".join(resultat["problemes"]))
        with st.expander("Diagnostic : réponses brutes du modèle"):
            for b in resultat.get("brouillons", []):
                st.caption(f"Rédaction {b['tentative']} · fin : {b['fin'] or 'non précisée'} · usage : {b['usage']}")
                st.code(b["texte"] or "(réponse vide)", language="markdown")
    else:
        essais = resultat["tentatives"]
        st.caption(f":material/verified: Contrôles automatiques réussis (sections, sources, montants, articles, "
                   f"confidentialité) · {essais} rédaction{'s' if essais > 1 else ''}.")
    with st.expander(f"Sources utilisées ({len(resultat['documents'])})"):
        for doc in resultat["documents"]:
            st.markdown(f"**[{doc.id}] {doc.titre}**  \n{doc.source}")


# ==========================================================================
# Page 3 : performance (mode démo)
# ==========================================================================
def page_performance():
    entete("Pilotage", "Performance de la détection",
           "Comparaison des trois classements à la vérité terrain des données synthétiques.")
    if not mode_demo():
        st.info("Activez le mode démo dans la barre latérale pour comparer les scores à la vérité terrain.")
        return
    scores = charger_scores()
    labels, _, profils = charger_verite_terrain()
    df = scores.merge(labels, on="numero_dossier", how="left")
    df["suspect"] = df["typologie"].notna()
    n = df["suspect"].sum()

    courbes = []
    for col, nom in SCORES.items():
        trouves = df.sort_values(col, ascending=False)["suspect"].cumsum().to_numpy()
        for k in range(10, 1001, 10):
            courbes.append({"Méthode": nom, "Dossiers examinés": k,
                            "Rappel": trouves[k - 1] / n, "Précision": trouves[k - 1] / k})
    courbes = pd.DataFrame(courbes)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(courbes, x="Dossiers examinés", y="Rappel", color="Méthode",
                      color_discrete_sequence=theme.METHODES)
        fig.update_layout(height=400, yaxis_tickformat=".0%", title="Part des suspects retrouvés",
                          legend=dict(orientation="h", y=-0.3, title=None), margin=dict(b=90))
        theme.styliser(fig)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(courbes, x="Dossiers examinés", y="Précision", color="Méthode",
                      color_discrete_sequence=theme.METHODES)
        fig.update_layout(height=400, yaxis_tickformat=".0%", title="Part de vrais suspects dans la file",
                          legend=dict(orientation="h", y=-0.3, title=None), margin=dict(b=90))
        theme.styliser(fig)
        st.plotly_chart(fig, width="stretch")

    k = st.slider("Taille de la file examinée", 50, 1000, 300, 50)
    top = df.sort_values("score_final", ascending=False).head(k)
    trouves = top[top["suspect"]]
    rappel = (trouves.groupby(["typologie", "intensite"]).size() / df[df["suspect"]].groupby(["typologie", "intensite"]).size()).fillna(0)
    st.markdown(f"**Avec {k} dossiers examinés** : {len(trouves)} suspects retrouvés sur {n} "
                f"({len(trouves) / n:.0%}), {k - len(trouves)} fausses alertes.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Rappel par typologie (hybride)")
        tableau = rappel.unstack().rename(columns={"discrete": "Discrète", "forte": "Forte"}).rename(index=NOMS_TYPOLOGIES)
        tableau.index.name = "Typologie"
        st.dataframe(tableau.style.format("{:.0%}"),
                     width="stretch")
    with c2:
        st.markdown("Fausses alertes : clients légitimes atypiques")
        faux = top.merge(profils, on="numero_dossier", how="inner")["profil"].map(NOMS_PROFILS).value_counts()
        faux.index.name = "Profil légitime"
        st.dataframe(faux.rename("Clients"), width="stretch")


# ==========================================================================
# Navigation
# ==========================================================================
def mode_demo() -> bool:
    return st.session_state.get("demo", False)


PAGE_ALERTES = st.Page(page_alertes, title="File d'alertes", icon=":material/notifications_active:", default=True)
PAGE_FICHE = st.Page(page_fiche, title="Fiche client", icon=":material/person_search:")
PAGE_PERF = st.Page(page_performance, title="Performance", icon=":material/monitoring:")


def controle_acces() -> bool:
    """Code d'accès optionnel (variable APP_CODE_ACCES) : protège une démo publique,
    et donc les appels à Gemini, des visiteurs non invités."""
    attendu = os.environ.get("APP_CODE_ACCES")
    if not attendu or st.session_state.get("acces_ok"):
        return True
    entete("Accès", "Analyste LCB-FT", "Démonstration sur données 100 % synthétiques.")
    with st.form("acces"):
        code = st.text_input("Code d'accès", type="password")
        if st.form_submit_button("Entrer", type="primary"):
            if hmac.compare_digest(code, attendu):
                st.session_state.acces_ok = True
                st.rerun()
            st.error("Code incorrect.")
    return False


acces_autorise = controle_acces()
navigation = st.navigation({"Surveillance": [PAGE_ALERTES, PAGE_FICHE], "Pilotage": [PAGE_PERF]},
                           position="sidebar" if acces_autorise else "hidden")
if acces_autorise:
    with st.sidebar:
        st.text_input("Analyste", key="analyste", placeholder="Votre nom", help="Inscrit dans le journal des décisions.")
        st.toggle("Mode démo", key="demo",
                  help="Affiche la vérité terrain des données synthétiques. Un analyste réel ne l'a pas.")
        export = journal.lire()
        st.download_button("Exporter le journal des décisions", export.to_csv(index=False).encode("utf-8"),
                           file_name="journal_decisions.csv", mime="text/csv", icon=":material/download:",
                           disabled=export.empty, width="stretch")
        st.caption("Données 100 % synthétiques")
        st.divider()
    navigation.run()
