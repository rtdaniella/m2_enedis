from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from components.navbar import create_sidenav
from components.footer import create_footer
from pages.home import create_home_page
from pages.about import create_about_page
from pages.context import create_context_page 
from pages.charts import create_charts_page  
from pages.map import create_map_page  
from pages.not_found_404 import create_not_found_page

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = create_sidenav()
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
        return create_map_page()  # Appelle la page Map
    else:
        return create_not_found_page()  # Appelle la page 404
    
    
if __name__ == '__main__':
    app.run_server(debug=True)
