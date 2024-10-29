from dash import html

def create_about_page():
    return html.Div([
        html.H1("À propos de cette application"),
        html.P("Ceci est la page d'informations à propos de l'application."),
    ])
