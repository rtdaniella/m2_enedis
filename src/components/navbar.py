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
                                dbc.NavLink("Home", href="/", className="nav-link text-white", active="exact"),
                                dbc.NavLink("Context", href="/context", className="nav-link text-white", active="exact"),
                                dbc.NavLink("Charts", href="/charts", className="nav-link text-white", active="exact"),
                                dbc.NavLink("Map", href="/map", className="nav-link text-white", active="exact"),
                                dbc.NavLink("About", href="/about", className="nav-link text-white", active="exact"),
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
