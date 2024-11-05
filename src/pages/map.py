from dash import html

def create_map_page():
    return html.Div([
        html.H1("Page Map"),
        html.P("Ceci est le contenu de la page Map."),
    ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        })
