from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash

# Pour l'icône X, tu peux utiliser l'icône de Font Awesome
def create_pred_dpe_page():
    return html.Div([
        dbc.Button(
            [
                html.I(className="fas fa-times me-2"),  # Icône X avec espace
                "Retour"
            ],
            id="back-button",
            color="danger",
            style={
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "zIndex": "999"
            },
        ),
        html.H1("Prediction DPE"),
        html.P("Ceci est la page de prédiction DPE."),
    ],
    style={
        "padding": "20px",
        "marginLeft": "260px"
    })
