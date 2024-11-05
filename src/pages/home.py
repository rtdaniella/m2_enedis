from dash import html

def create_home_page():
    return html.Div(
        [
            # Bloc principal en haut
            html.Div(
                [
                    html.H3("Ceci est le contenu de la page d'accueil.", style={"textAlign": "center"}),
                    html.Div(
                        [
                            html.Button(
                                "Prédiction étiquette DPE",
                                style={
                                    "margin": "5px",
                                    "padding": "10px 20px",
                                    "backgroundColor": "#007BFF",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "transition": "background-color 0.3s, transform 0.3s",
                                },
                                id="button-1"
                            ),
                            html.Button(
                                "Prédiction consommation énérgétique",
                                style={
                                    "margin": "5px",
                                    "padding": "10px 20px",
                                    "backgroundColor": "#28A745",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "transition": "background-color 0.3s, transform 0.3s",
                                },
                                id="button-2"
                            ),
                        ],
                        style={"textAlign": "center", "marginTop": "20px"}
                    )
                ],
                style={
                    "backgroundColor": "lightgrey",
                    "padding": "20px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                    "marginBottom": "20px"
                }
            ),
            # Deux blocs horizontaux en bas
            html.Div(
                [
                    html.Div(
                        "Contenu du bloc gauche",
                        style={
                            "backgroundColor": "lightgrey",
                            "padding": "20px",
                            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                            "flex": "1",
                            "marginRight": "10px"  # Espace entre les deux blocs
                        }
                    ),
                    html.Div(
                        "Contenu du bloc droit",
                        style={
                            "backgroundColor": "lightgrey",
                            "padding": "20px",
                            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                            "flex": "1"
                        }
                    )
                ],
                style={
                    "display": "flex"  # Disposition horizontale
                }
            )
        ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        }
    )
