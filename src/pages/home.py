from dash import html

def create_home_page():
    return html.Div([
        html.H1("Bienvenue sur la page d'accueil"),
        html.P("Ceci est le contenu de la page d'accueil."),
    ])
