from dash import html

def create_context_page():
    return html.Div([
        html.H1("Page Context"),
        html.P("Ceci est le contenu de la page Context."),
    ])
