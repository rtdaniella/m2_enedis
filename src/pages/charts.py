from functools import lru_cache
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html


@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv("src/files/dpe-nettoye.csv")


def create_charts_page():
    # Chargement des données
    df = load_data()

    # Liste des colonnes pour le boxplot avec labels plus lisibles
    options_boxplot = [
        {"label": "Mode de chauffage", "value": "type_installation_chauffage"},
        {"label": "Type d'énergie", "value": "type_energie_n_1"},
        {"label": "Type de logement", "value": "logement"},
        {"label": "Période de construction", "value": "periode_construction"},
        {"label": "Statut 'passoire énergétique'", "value": "passoire_energetique"},
    ]

    options_piechart = [
        {"label": "Type bâtiment", "value": "type_batiment"},
        {"label": "Période de construction", "value": "periode_construction"},
        {"label": "Type de logement", "value": "logement"},
    ]

    options_scatterplt = [
        {"label": "A", "value": "A"},
        {"label": "B", "value": "B"},
        {"label": "C", "value": "C"},
        {"label": "D", "value": "D"},
        {"label": "E", "value": "E"},
        {"label": "F", "value": "F"},
        {"label": "G", "value": "G"},
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
                            # Graphique 1 : Nombre et type de DPE réalisés chaque mois
                            html.Div(
                                [
                                    html.H3(
                                        "Nombre et type de DPE réalisés chaque mois"
                                    ),
                                    dcc.Graph(id="dpe_par_mois"),
                                ],
                                className="col-md-12",
                            ),
                        ],
                        className="row mb-4",
                    ),
                    # Deuxième ligne
                    html.Div(
                        [
                            # Graphique 2: Distribution des étiquettes DP
                            html.Div(
                                [
                                    html.H3("Répartition des étiquettes DPE"),
                                    dcc.Graph(id="distribution_dpe"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 3: Distribution des surfaces
                            html.Div(
                                [
                                    html.H3("Distribution des surfaces des logements"),
                                    dcc.Graph(id="distribution_surf"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 3: Distribution des types de logement
                            html.Div(
                                [
                                    html.H3("Répartition des types de logement"),
                                    # Boutons radio pour choisir la variable de regroupement
                                    dcc.RadioItems(
                                        id="pie_grouping",
                                        options=options_piechart,
                                        value="logement",
                                        inline=True,  # Disposition horizontale
                                        className="mb-3",  # Marge en bas
                                        inputStyle={
                                            "margin-right": "5px"
                                        },  # Espace entre le bouton et le label
                                        labelStyle={
                                            "margin-right": "15px"
                                        },  # Espace entre les options
                                    ),
                                    dcc.Graph(id="repartition_logement"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 4: scatter plot
                            html.Div(
                                [
                                    html.H3(
                                        "Scatter plot surface / consommation par étiquette DPE"
                                    ),
                                    # Boutons radio pour choisir la variable de regroupement
                                    dcc.RadioItems(
                                        id="scatter_grouping",
                                        options=options_scatterplt,
                                        value="A",
                                        inline=True,  # Disposition horizontale
                                        className="mb-3",  # Marge en bas
                                        inputStyle={
                                            "margin-right": "5px"
                                        },  # Espace entre le bouton et le label
                                        labelStyle={
                                            "margin-right": "15px"
                                        },  # Espace entre les options
                                    ),
                                    dcc.Graph(id="scatter_surface_conso"),
                                ],
                                className="col-md-6",
                            ),
                            # Graphique 4: Boîte à moustaches
                            html.Div(
                                [
                                    html.H3("Consommation annuelle par..."),
                                    # Boutons radio pour choisir la variable de regroupement
                                    dcc.RadioItems(
                                        id="boxplot_grouping",
                                        options=options_boxplot,
                                        value="type_installation_chauffage",
                                        inline=True,  # Disposition horizontale
                                        className="mb-3",  # Marge en bas
                                        inputStyle={
                                            "margin-right": "5px"
                                        },  # Espace entre le bouton et le label
                                        labelStyle={
                                            "margin-right": "15px"
                                        },  # Espace entre les options
                                    ),
                                    dcc.Graph(id="bam_conso"),
                                ],
                                className="col-md-12",
                            ),
                        ],
                        className="row mb-4",
                    ),
                ],
                className="container-fluid",
            ),
        ],
        className="p-4",
        style={
            "padding": "20px",
            "marginLeft": "260px",  # Décalage pour laisser de l'espace à la SideNav
        },
    )

    return layout


@callback(Output("dpe_par_mois", "figure"), Input("dpe_par_mois", "id"))
def update_dpe_par_mois(_):
    df = load_data()

    counts_df = (
        df.groupby(["yearmonth", "etiquette_dpe"]).size().reset_index(name="count")
    )

    # Définition des couleurs par catégorie DPE
    color_map_dpe = {
        "A": "#008f35",
        "B": "#57aa28",
        "C": "#c8d200",
        "D": "#fcea26",
        "E": "#f8bb00",
        "F": "#ea690b",
        "G": "#e30c1c",
    }

    fig = px.bar(
        counts_df,
        x="yearmonth",
        y="count",
        color="etiquette_dpe",  # Add this to enable coloring by DPE category
        color_discrete_map=color_map_dpe,  # Use the predefined color map
        # title="Distribution des catégories par mois",
        labels={
            "yearmonth": "Date",
            "count": "Nombre d'enregistrements",
            "etiquette_dpe": "Etiquette DPE",
        },
    )

    # Personnalisation du graphique
    fig.update_layout(
        barmode="stack",
        xaxis_tickangle=-45,
        bargap=0.1,
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Ajout d'une grille
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey")

    return fig


@callback(Output("repartition_logement", "figure"), Input("pie_grouping", "value"))
def update_housing_distribution(grouping_column):
    df = load_data()
    counts_df = df[grouping_column].value_counts().reset_index()
    counts_df.columns = [grouping_column, "count"]

    fig = px.pie(
        counts_df,
        values="count",
        names=grouping_column,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=50, l=0, r=0, b=0))
    return fig


@callback(Output("distribution_dpe", "figure"), Input("distribution_dpe", "id"))
def update_dpe_distribution(_):
    df = load_data()
    dpe_order = ["A", "B", "C", "D", "E", "F", "G"]
    dpe_counts = df["etiquette_dpe"].value_counts().reindex(dpe_order).reset_index()
    color_map_dpe = {
        "A": "#008f35",
        "B": "#57aa28",
        "C": "#c8d200",
        "D": "#fcea26",
        "E": "#f8bb00",
        "F": "#ea690b",
        "G": "#e30c1c",
    }

    fig = px.bar(
        dpe_counts,
        x="etiquette_dpe",
        y="count",
        template="plotly_white",
        color="etiquette_dpe",
        color_discrete_map=color_map_dpe,
    )

    fig.update_layout(
        xaxis_title="Étiquette DPE",
        yaxis_title="Nombre de logements",
        showlegend=False,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    fig.update_traces(texttemplate="%{y}", textposition="outside")
    return fig


@callback(Output("distribution_surf", "figure"), Input("distribution_surf", "id"))
def update_surf_distribution(_):
    df = load_data()

    fig = px.histogram(
        df,
        x="surface_habitable_logement",
        nbins=30,
        # title="Distribution des surfaces habitables",
        template="plotly_white",
        hover_data={"surface_habitable_logement": ":.0f"},
    )

    fig.update_layout(
        xaxis_title="Surface habitable (m²)",
        yaxis_title="Nombre de logements",
        showlegend=False,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    return fig


@callback(Output("bam_conso", "figure"), Input("boxplot_grouping", "value"))
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
        # Conversion de la colonne en type catégoriel avec l'ordre spécifié
        df[grouping_column] = pd.Categorical(
            df[grouping_column], categories=periodes, ordered=True
        )
        # Tri du DataFrame
        df = df.sort_values(grouping_column)

    fig = px.box(
        df,
        x=grouping_column,
        y="conso_5_usages_e_finale",
        # title=f'Consommation annuelle par {grouping_column.replace("_", " ").title()}',
        template="plotly_white",
        color=grouping_column,
        color_discrete_sequence=px.colors.qualitative.Set2,
        # points=False,  # Enlever les outliers
    )

    # Personnalisation du titre de l'axe X selon la variable choisie
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


@callback(Output("scatter_surface_conso", "figure"), Input("scatter_grouping", "value"))
def update_scatter_surface_conso(grouping_value):
    df = load_data()

    # Filtrer les données en fonction de la valeur sélectionnée
    filtered_df = df[df["etiquette_dpe"] == grouping_value]

    fig = px.scatter(
        filtered_df,
        x="surface_habitable_logement",
        y="conso_5_usages_e_finale",
        # title="Relation entre la surface habitable et la consommation énergétique",
        template="plotly_white",
        color="etiquette_dpe",
        color_discrete_map={
            "A": "#008f35",
            "B": "#57aa28",
            "C": "#c8d200",
            "D": "#fcea26",
            "E": "#f8bb00",
            "F": "#ea690b",
            "G": "#e30c1c",
        },
    )

    fig.update_layout(
        xaxis_title="Surface habitable (m²)",
        yaxis_title="Consommation en kWhef/an",
        showlegend=True,
        margin=dict(t=50, l=50, r=0, b=50),
    )

    # Mettre à jour les plages des axes en fonction des valeurs min et max
    fig.update_xaxes(
        range=[
            0,
            150,
        ]
    )
    fig.update_yaxes(
        range=[
            0,
            30000,
            # filtered_df["conso_5_usages_e_finale"].max(),
        ]
    )

    return fig
