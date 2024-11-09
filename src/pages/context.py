from functools import lru_cache
import pandas as pd
from dash import html, dash_table, dcc, Input, Output, callback, State
import dash
from datetime import datetime
import numpy as np

# Constantes pour la pagination
PAGE_SIZE = 15
INITIAL_PAGE = 0


@lru_cache(maxsize=1)
def load_data():
    # Optimisation du chargement initial en utilisant des types de données optimisés
    dtype_dict = {
        "code_postal_ban": "category",
        "etiquette_dpe": "category",
        "logement": "category",
        "passoire_energetique": "category",
        "type_batiment": "category",
        "type_energie_n_1": "category",
        "type_installation_chauffage": "category",
        "periode_construction": "category",
        "yearmonth": "category",
        "surface_habitable_logement": "float32",
        "cout_total_5_usages": "float32",
        "conso_5_usages_e_finale": "float32",
    }

    return pd.read_csv(
        "src/files/dpe-nettoye.csv",
        usecols=[
            "code_postal_ban",
            "etiquette_dpe",
            "cout_total_5_usages",
            "conso_5_usages_e_finale",
            "logement",
            "passoire_energetique",
            "type_batiment",
            "type_energie_n_1",
            "type_installation_chauffage",
            "periode_construction",
            "yearmonth",
            "surface_habitable_logement",
        ],
        dtype=dtype_dict,
    )


@lru_cache(maxsize=1)
def get_unique_values():
    data = load_data()
    # Optimisation de la récupération des valeurs uniques
    return {
        "dpe": data["etiquette_dpe"].cat.categories.tolist(),
        "construction": data["periode_construction"].cat.categories.tolist(),
        "energie": data["type_energie_n_1"].cat.categories.tolist(),
        "postal": data["code_postal_ban"].cat.categories.tolist(),
    }


def calculate_kpis(data):
    return {
        "total_entries": len(data),
        "average_dpe": data["etiquette_dpe"].mode().iloc[0],
        "total_postals": data["code_postal_ban"].nunique(),
    }


def create_context_page():
    data = load_data()
    unique_values = get_unique_values()
    kpi = calculate_kpis(data)

    layout = html.Div(
        [
            # En-tête avec lazy loading
            dcc.Loading(
                id="loading-header",
                children=[
                    html.Div(
                        [
                            html.I(
                                className="fas fa-chart-line",
                                style={
                                    "fontSize": "36px",
                                    "color": "white",
                                    "marginRight": "15px",
                                    "verticalAlign": "middle",
                                },
                            ),
                            html.H1(
                                "Contexte",
                                style={
                                    "color": "white",
                                    "fontSize": "32px",
                                    "fontWeight": "bold",
                                    "display": "inline-block",
                                    "marginBottom": "0",
                                },
                            ),
                        ],
                        style={
                            "background": "linear-gradient(135deg, #2a6cb2, #1a3d63)",
                            "padding": "20px",
                            "borderRadius": "15px",
                            "boxShadow": "0 4px 10px rgba(0, 0, 0, 0.1)",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "marginBottom": "20px",
                            "marginLeft": "10px",
                            "marginTop": "10px",
                        },
                    ),
                ],
                type="default",
            ),
            # KPI Cards avec Lazy Loading
            dcc.Loading(
                id="loading-kpis",
                children=[
                    html.Div(
                        [
                            create_kpi_card(
                                "fas fa-house-user",
                                "Total des Entrées",
                                kpi["total_entries"],
                            ),
                            create_kpi_card(
                                "fas fa-calendar-alt",
                                "DPE le plus courant",
                                kpi["average_dpe"],
                            ),
                            create_kpi_card(
                                "fas fa-map-marker-alt",
                                "Total des Codes Postaux",
                                kpi["total_postals"],
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "gap": "20px",
                            "marginBottom": "30px",
                        },
                    )
                ],
                type="default",
            ),
            # Bouton d'exportation
            html.Div(
                [
                    html.Button(
                        "Exporter la sélection filtrée (CSV)",
                        id="btn-export",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#2a6cb2",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 20px",
                            "textAlign": "center",
                            "textDecoration": "none",
                            "display": "inline-block",
                            "fontSize": "16px",
                            "margin": "4px 2px",
                            "cursor": "pointer",
                            "borderRadius": "10px",
                        },
                    ),
                    dcc.Download(id="download-dataframe-csv"),
                ],
                style={"textAlign": "center", "marginTop": "20px"},
            ),
            # Filtres optimisés
            html.Div([create_filter_section(unique_values)]),
            # DataTable avec pagination côté serveur
            dcc.Loading(
                id="loading-table",
                children=[
                    dash_table.DataTable(
                        id="data-table",
                        columns=[{"name": col, "id": col} for col in data.columns],
                        page_size=PAGE_SIZE,
                        page_current=INITIAL_PAGE,
                        page_action="custom",
                        sort_action="custom",
                        sort_mode="single",
                        filter_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "5px"},
                        style_header={
                            "backgroundColor": "grey",
                            "fontWeight": "bold",
                            "color": "white",
                        },
                    )
                ],
                type="default",
            ),
        ],
        style={
            "padding": "20px",
            "marginLeft": "260px",
            "marginBottom": "75px",
        },
    )

    return layout


def create_kpi_card(icon_class, title, value):
    return html.Div(
        [
            html.I(className=icon_class, style={"color": "white"}),
            html.H5(title, style={"color": "white", "marginTop": "10px"}),
            html.P(
                str(value),
                style={"color": "white", "fontSize": "24px", "fontWeight": "bold"},
            ),
        ],
        style={
            "borderRadius": "10px",
            "padding": "20px",
            "width": "250px",
            "textAlign": "center",
            "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
            "backgroundColor": "#2a6cb2",
        },
    )


def create_filter_section(unique_values):
    return html.Div(
        [
            html.Div(
                [
                    create_filter(
                        "dpe-filter", "Filtrer par étiquette DPE", unique_values["dpe"]
                    ),
                    create_filter(
                        "periode-construction-filter",
                        "Période de construction",
                        unique_values["construction"],
                    ),
                ],
                style={"width": "48%", "marginRight": "4%"},
            ),
            html.Div(
                [
                    create_filter(
                        "type-energie-filter",
                        "Type d'énergie",
                        unique_values["energie"],
                    ),
                    create_filter(
                        "code-postal-filter", "Code postal", unique_values["postal"]
                    ),
                ],
                style={"width": "48%"},
            ),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "marginBottom": "20px",
        },
    )


def create_filter(id, label, options):
    return html.Div(
        [
            html.Label(label, style={"fontWeight": "bold", "marginBottom": "10px"}),
            dcc.Dropdown(
                id=id,
                options=[{"label": str(val), "value": val} for val in options],
                multi=True,
                placeholder=f"Sélectionner {label}",
                style={"width": "100%"},
                clearable=True,
            ),
        ],
        style={"marginBottom": "20px"},
    )


@callback(
    Output("data-table", "data"),
    [
        Input("data-table", "page_current"),
        Input("data-table", "page_size"),
        Input("data-table", "sort_by"),
        Input("dpe-filter", "value"),
        Input("periode-construction-filter", "value"),
        Input("type-energie-filter", "value"),
        Input("code-postal-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_table(
    page_current,
    page_size,
    sort_by,
    dpe_filter,
    periode_filter,
    energie_filter,
    postal_filter,
):
    # Chargement des données avec cache
    df = load_data()

    # Application des filtres
    if dpe_filter:
        df = df[df["etiquette_dpe"].isin(dpe_filter)]
    if periode_filter:
        df = df[df["periode_construction"].isin(periode_filter)]
    if energie_filter:
        df = df[df["type_energie_n_1"].isin(energie_filter)]
    if postal_filter:
        df = df[df["code_postal_ban"].isin(postal_filter)]

    # Tri
    if sort_by:
        df = df.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            na_position="last",
        )

    # Pagination
    start = page_current * page_size
    end = start + page_size

    return df.iloc[start:end].to_dict("records")


@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-export", "n_clicks"),
    [
        State("dpe-filter", "value"),
        State("periode-construction-filter", "value"),
        State("type-energie-filter", "value"),
        State("code-postal-filter", "value"),
    ],
    prevent_initial_call=True,
)
def export_data(n_clicks, dpe_filter, periode_filter, energie_filter, postal_filter):
    if not n_clicks:
        return None

    df = load_data()

    # Application des filtres
    if dpe_filter:
        df = df[df["etiquette_dpe"].isin(dpe_filter)]
    if periode_filter:
        df = df[df["periode_construction"].isin(periode_filter)]
    if energie_filter:
        df = df[df["type_energie_n_1"].isin(energie_filter)]
    if postal_filter:
        df = df[df["code_postal_ban"].isin(postal_filter)]

    return dcc.send_data_frame(
        df.to_csv,
        f"export_donnees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )
