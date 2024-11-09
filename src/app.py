from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

from components.navbar import create_navbar
from components.footer import create_footer
from pages.home import create_home_page
from pages.about import create_about_page
from pages.context import create_context_page 
from pages.charts import create_charts_page  
from pages.map import create_map_page, register_callbacks  
from pages.not_found_404 import create_not_found_page
from pages.predict import create_prediction_page, register_prediction_callbacks

# Charger les données d'entrée
dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

navbar = create_navbar()
footer = create_footer()

# Layout de l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Permet de détecter l'URL
    navbar,
    html.Div(id="page-content"),  # Contenu de la page sélectionnée
    footer
])

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
        return create_map_page(dfa, dfn)  # Appelle la page Map
    elif pathname == "/predict":  # Nouvelle route pour la page de prédiction
        return create_prediction_page()  # Appelle la page de prédiction
    else:
        return create_not_found_page()  # Appelle la page 404

# Enregistrer les callbacks pour la page de prédiction
register_prediction_callbacks(app)  # Fonction spécifique pour enregistrer les callbacks de prédiction

# Enregistrer les callbacks pour la page de la carte
register_callbacks(app, dfa, dfn)  # Fonction spécifique pour enregistrer les callbacks de la carte

if __name__ == '__main__':
    app.run_server(debug=True)
