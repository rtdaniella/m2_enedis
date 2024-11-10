import dash_bootstrap_components as dbc
from dash import html

# SideNav Definition
def create_sidenav():
    sidenav = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                # Colonne pour le logo
                                dbc.Col(
                                    html.Img(
                                        src="/assets/images/logo.png",  # Chemin vers le fichier du logo
                                        style={
                                            "height": "80px",
                                            "width": "80px",
                                            "marginRight": "10px",
                                            "marginLeft": "70px",
                                            "marginTop": "40px"
                                        }
                                    ),
                                    width="auto"
                                ),
                                # Colonne pour le texte à côté du logo
                                dbc.Col(
                                    html.H2("GreenTech Solutions", className="text-light mt-1"),
                                    style={
                                        "marginTop": "10px",
                                        "textAlign": "center"
                                    }
                                ),
                            ],
                            align="center",  # Alignement vertical au centre
                            className="mb-3"  # Marge en bas pour séparer du reste
                        ),
                        html.Hr(),
                        dbc.Nav(
                            [
                                dbc.NavLink(
                                    [html.I(className="fas fa-home me-2"), "Accueil"],  # Icône pour "Home"
                                    href="/",
                                    className="nav-link text-white",
                                    active="exact",
                                ),
                                dbc.NavLink(
                                    [html.I(className="fas fa-table me-2"), "Contexte"],  # Icône pour "Context"
                                    href="/context",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                                dbc.NavLink(
                                    [html.I(className="fas fa-chart-bar me-2"), "Graphiques"],  # Icône pour "Charts"
                                    href="/charts",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                                dbc.NavLink(
                                    [html.I(className="fas fa-map-marker-alt me-2"), "Carte"],  # Icône pour "Map"
                                    href="/map",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                                dbc.NavLink(
                                    [html.I(className="fas fa-question-circle me-2"), "A propos"],  # Icône pour "About"
                                    href="/about",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                                html.Div(style={"height": "10px"}),  # Espacement entre "A propos" et la ligne de séparation
                                
                                # Trait séparateur
                                html.Div(
                                    style={
                                        "borderTop": "2px solid white",  # Ligne de séparation blanche
                                        "marginTop": "20px",
                                        "marginBottom": "10px"
                                    }
                                ),
                                
                                # Sous-menus "Predict DPE" et "Predict Conso" affichés par défaut
                                dbc.NavLink(
                                    [html.I(className="fas fa-cogs me-2"), "Etiquette DPE"],
                                    href="/pred_dpe",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                                dbc.NavLink(
                                    [html.I(className="fas fa-cogs me-2"), "Consommation énergétique"],
                                    href="/pred_conso",
                                    className="nav-link text-white",
                                    active="exact"
                                ),
                            ],
                            vertical=True,
                            pills=True,
                            className="mt-4"
                        ),
                    ],
                    style={
                        'background': 'linear-gradient(135deg, #000428, #004e92)',
                        "height": "100vh",
                        "width": "250px",
                        "position": "fixed",
                        "padding": "15px",
                        "boxShadow": "5px 0 10px rgba(0, 0, 0, 0.5)",
                    },
                )
            ),
        ],
        fluid=True
    )
    return sidenav
