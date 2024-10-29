from dash import html

def create_not_found_page():
    return html.Div([
        html.H1("404: Page not found"),
        html.P("Désolé, cette page n'existe pas."),
    ])
