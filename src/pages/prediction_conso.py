from dash import html

def create_pred_conso_page():
    return html.Div([
        html.H1("Prediction conso"),
        html.P("Ceci est la page d'informations à propos de l'application."),
    ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        })