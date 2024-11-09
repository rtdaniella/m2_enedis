from functools import lru_cache
from dash import html
import pandas as pd
from dash import Input, Output, callback, dcc, html
import plotly.graph_objects as go


@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv("src/files/dpe-nettoye.csv")


def create_map_page():

    df = load_data()

    layout = html.Div(
        [
            # Bloc du titre avec fond en dégradé bleu vif à bleu foncé
            html.Div(
                [
                    html.I(
                        className="fas fa-globe-americas",  # Icône Font Awesome
                        style={
                            "fontSize": "36px",
                            "color": "#ffffff",  # Icône blanche
                            "marginRight": "15px",
                            "verticalAlign": "middle",
                        },
                    ),
                    html.H1(
                        "Carte Interactive",
                        style={
                            "color": "#ffffff",  # Couleur du texte en blanc
                            "fontSize": "32px",  # Taille de police plus grande
                            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",  # Typographie moderne
                            "fontWeight": "bold",
                            "display": "inline-block",  # Affichage en bloc pour centrer le texte
                            "lineHeight": "1.2",  # Un peu plus d'espace entre les lignes
                            "textAlign": "center",  # Centrer le texte
                            "marginBottom": "0",  # Pas de marge en bas pour centrer l'élément
                        },
                    ),
                ],
                style={
                    "background": "linear-gradient(135deg, #2a6cb2, #1a3d63)",  # Dégradé bleu plus profond et moins clair
                    "padding": "20px",
                    "borderRadius": "15px",  # Coins arrondis
                    "boxShadow": "0 4px 10px rgba(0, 0, 0, 0.1)",  # Ombre légère pour donner de la profondeur
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",  # Centrer le contenu horizontalement
                    "marginBottom": "30px",
                    "marginLeft": "10px",  # Décalage pour la SideNav
                    "marginTop": "10px",
                    "marginRight": "10px",
                },
            ),
            dcc.RadioItems(
                id="dpe-filter",
                options=[
                    {"label": etiq, "value": etiq}
                    for etiq in sorted(df["etiquette_dpe"].unique())
                ],
                value="A",
                labelStyle={"display": "inline-block", "margin-right": "10px"},
            ),
            html.Div(id="map-container"),
        ],
        style={
            "padding": "20px",
            "marginLeft": "260px",  # Décalage pour laisser de l'espace à la SideNav
        },
    )
    return layout


@callback(Output("map-container", "children"), [Input("dpe-filter", "value")])
def update_map(selected_dpe):
    df = load_data()
    filtered_df = df[df["etiquette_dpe"] == selected_dpe]
    color_map_dpe = {
        "A": "#008f35",
        "B": "#57aa28",
        "C": "#c8d200",
        "D": "#fcea26",
        "E": "#f8bb00",
        "F": "#ea690b",
        "G": "#e30c1c",
    }

    fig = go.Figure(
        go.Scattermapbox(
            lat=filtered_df["latitude"],
            lon=filtered_df["longitude"],
            mode="markers",
            marker=dict(
                size=10,
                color=[color_map_dpe[etiq] for etiq in filtered_df["etiquette_dpe"]],
                symbol="circle",
            ),
            text=[f"DPE: {etiq}" for etiq in filtered_df["etiquette_dpe"]],
        ),
        layout=dict(
            uirevision="constant",  # This maintains the zoom/position state
            mapbox=dict(
                style="open-street-map",
                zoom=12,
                center={"lat": 45.7440, "lon": 4.8457},
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=700,
        ),
    )

    # Retourner le Graph avec l'option scrollZoom activée
    return dcc.Graph(
        figure=fig,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )
