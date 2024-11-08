from dash import dcc, html, Input, Output, callback
from functools import lru_cache
import pandas as pd
import plotly.express as px

@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv("C:/Users/danie/OneDrive/Documents/GitHub/m2_enedis/src/files/dpe-nettoye.csv")


def create_charts_page():
    """Crée la page des graphiques avec une disposition en grille 2x2"""

    # Chargement des données
    df = load_data()

    # Liste des colonnes pour le boxplot avec labels plus lisibles
    options_boxplot = [
        {"label": "Passoire énergétique", "value": "passoire_energetique"},
        {"label": "Période de construction", "value": "periode_construction"},
        {"label": "Type de logement", "value": "logement"},
    ]

    # Création du layout
    layout = html.Div(
        [
            html.H1(
                "Analyse des données DPE sur le département du Rhône",
                className="text-center mb-4",
            ),
            # Grille 2x2 pour les graphiques
            html.Div(
                [
                    # Première ligne
                    html.Div(
                        [
                            # Graphique 1: Distribution des types de logement
                            html.Div(
                                [
                                    html.H3("Répartition des types de logement"),
                                    dcc.Graph(id="repartition_logement"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 2: Distribution des étiquettes DPE
                            html.Div(
                                [
                                    html.H3("Répartition des étiquettes DPE"),
                                    dcc.Graph(id="distribution_dpe"),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row mb-4",
                    ),
                    # Deuxième ligne
                    html.Div(
                        [
                            # Graphique 3: Nombre de DPE réalisés chaque mois
                            html.Div(
                                [
                                    html.H3("Nombre de DPE réalisés chaque mois"),
                                    dcc.Graph(id="dpe_monthly"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 4: Boîte à moustaches
                            html.Div(
                                [
                                    html.H3("Consommation annuelle par..."),
                                    # Boutons radio pour choisir la variable de regroupement
                                    dcc.RadioItems(
                                        id="boxplot-grouping",
                                        options=options_boxplot,
                                        value="passoire_energetique",
                                        inline=True,  # Disposition horizontale
                                        className="mb-3",  # Marge en bas
                                        inputStyle={
                                            "margin-right": "5px"
                                        },  # Espace entre le bouton et le label
                                        labelStyle={
                                            "margin-right": "15px"
                                        },  # Espace entre les options
                                    ),
                                    dcc.Graph(id="cost-boxplot"),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row mb-4",
                    ),
                ],
                className="container-fluid",
            ),
        ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        },
    )

    return layout


@callback(Output("repartition_logement", "figure"), Input("repartition_logement", "id"))
def update_housing_distribution(_):
    df = load_data()
    fig = px.pie(
        df["logement"].value_counts().reset_index(),
        values="count",
        names="logement",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
    return fig


@callback(Output("distribution_dpe", "figure"), Input("distribution_dpe", "id"))
def update_dpe_distribution(_):
    df = load_data()
    dpe_order = ["A", "B", "C", "D", "E", "F", "G"]
    dpe_counts = df["etiquette_dpe"].value_counts().reindex(dpe_order).reset_index()

    fig = px.bar(
        dpe_counts,
        x="etiquette_dpe",
        y="count",
        template="plotly_white",
        color="etiquette_dpe",
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    fig.update_layout(
        xaxis_title="Étiquette DPE",
        yaxis_title="Nombre de logements",
        showlegend=False,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    fig.update_traces(texttemplate="%{y}", textposition="outside")
    return fig


@callback(Output("cost-boxplot", "figure"), Input("boxplot-grouping", "value"))
def update_cost_boxplot(grouping_column):
    df = load_data()

    # Définition de l'ordre pour la période de construction
    if grouping_column == "periode_construction":
        periodes = [
            "Inconnue",
            "Avant 1960",
            "1961 - 1970",
            "1971 - 1980",
            "1981 - 1990",
            "1991 - 2000",
            "2001 - 2010",
            "2011 - 2020",
            "Après 2020",
        ]
        df[grouping_column] = pd.Categorical(
            df[grouping_column], categories=periodes, ordered=True
        )
        df = df.sort_values(grouping_column)

    fig = px.box(
        df,
        x=grouping_column,
        y="conso_5_usages_e_finale",
        template="plotly_white",
        color=grouping_column,
        color_discrete_sequence=px.colors.qualitative.Set2,
        points=False,  # Enlever les outliers
    )

    x_title = {
        "passoire_energetique": "Passoire énergétique",
        "periode_construction": "Période de construction",
        "logement": "Type de logement",
    }.get(grouping_column, grouping_column.replace("_", " ").title())

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Consommation en kWhef/an",
        showlegend=False,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    return fig


@callback(Output("dpe_monthly", "figure"), Input("dpe_monthly", "id"))
def update_dpe_monthly(_):
    df = load_data()
    df["timestamp_reception_dpe"] = pd.to_datetime(
        df["timestamp_reception_dpe"], unit="s"
    )
    df["year_month"] = df["timestamp_reception_dpe"].dt.to_period("M").astype(str)

    fig = px.bar(
        df.groupby("year_month").size().reset_index(name="count"),
        x="year_month",
        y="count",
        template="plotly_white",
        color_discrete_sequence=["#4C78A8"],
    )

    fig.update_layout(
        xaxis_title="Année et Mois",
        yaxis_title="Nombre de DPE",
        showlegend=False,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    return fig
