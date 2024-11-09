from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from components.navbar import create_sidenav
from components.footer import create_footer
from pages.home import create_home_page
from pages.about import create_about_page
from pages.context import create_context_page 
from pages.charts import create_charts_page  
from pages.map import create_map_page, register_callbacks  
from pages.not_found_404 import create_not_found_page
# Importer les pages de prédiction
from pages.prediction_dpe import create_pred_dpe_page
from pages.prediction_conso import create_pred_conso_page

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
)

navbar = create_sidenav()
footer = create_footer()

# Layout de l'application
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),  # Permet de détecter l'URL
        navbar,
        html.Div(id="page-content"),  # Contenu de la page sélectionnée
        footer,
    ]
)


# Callback pour afficher la page en fonction de l'URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return create_home_page()  # Appelle la page d'accueil
    elif pathname == "/about":
        return create_about_page()  # Appelle la page "À propos"
    elif pathname == "/context":
        return create_context_page()  # Appelle la page Context
    elif pathname == "/charts":
         return create_charts_page()  # Appelle la page Charts
    elif pathname == "/map":
         return create_map_page()  # Appelle la page Map 
    elif pathname == "/pred_dpe":
        return create_pred_dpe_page()  # Appelle la page prédiction étiquette DPE
    elif pathname == "/pred_conso":
        return (
            create_pred_conso_page()
        )  # Appelle la page prédiction consommation énergétique
    else:
        return create_not_found_page()  # Appelle la page 404

# Enregistrer les callbacks pour la page de prédiction
register_prediction_callbacks(app)  # Fonction spécifique pour enregistrer les callbacks de prédiction

# Callback pour rediriger les boutons vers les pages de prédiction
@app.callback(
    Output("url", "pathname"),  # On modifie l'URL
    [
        Input("button-1", "n_clicks"),
        Input("button-2", "n_clicks"),
    ],  # Suivi des clics sur les boutons
)
def update_url(button1_clicks, button2_clicks):
    if button1_clicks:  # Si le bouton "Prédiction étiquette DPE" est cliqué
        return "/pred_dpe"
    elif (
        button2_clicks
    ):  # Si le bouton "Prédiction consommation énergétique" est cliqué
        return "/pred_conso"
    return "/"  # Si aucun bouton n'est cliqué, on reste sur la page d'accueil


if __name__ == "__main__":
    app.run_server(debug=False)
