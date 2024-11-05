from dash import html

def create_charts_page():
    return html.Div([
        html.H1("Page Charts"),
        html.P("Ceci est le contenu de la page Charts."),
    ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        })
