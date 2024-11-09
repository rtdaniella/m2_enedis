from dash import html

def create_home_page():
    return html.Div(
        [
            # Bloc principal en haut
            html.Div(
                [
                    html.H3("Bienvenue chez GreenTech Solutions.", style={"textAlign": "center", "marginTop":"30px"}),
                    html.P(
                        "À l'ère du changement climatique et de la hausse des prix de l’énergie, "
                        "la maîtrise de la consommation énergétique devient essentielle. "
                        "GreenTech Solutions est là pour vous accompagner grâce à des outils innovants "
                        "permettant de prédire et de mieux comprendre vos consommations d’énergie.",
                        style={"textAlign": "center", "margin": "40px 30px 30px 30px"}
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Prédiction étiquette DPE",
                                style={
                                    "margin": "5px",
                                    "padding": "10px 20px",
                                    "backgroundColor": "#56b536",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "transition": "background-color 0.3s, transform 0.3s",
                                },
                                id="button-1"
                            ),
                            html.Button(
                                "Prédiction consommation énergétique",
                                style={
                                    "margin": "5px",
                                    "padding": "10px 20px",
                                    "backgroundColor": "#56b536",
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
                    "padding": "10px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                    "marginBottom": "20px"
                }
            ),
            # Deux blocs horizontaux en bas
            html.Div(
                [
                    html.Div(
                        # Texte ajouté dans le bloc gauche
                        [
                            html.H4("Comprendre le Diagnostic de Performance Energétique (DPE)", style={"textAlign": "center"}),
                            html.P(
                                "Le Diagnostic de Performance Energétique (DPE) est un outil essentiel "
                                "qui évalue la consommation d’énergie d’un bâtiment et son impact environnemental. "
                                "Il attribue une étiquette allant de A (très performant) à G (peu performant). ",
                                style={"textAlign": "center", "marginTop": "20px"}
                            ),
                            # Ajouter une image sous le texte
                            html.Img(
                                src="assets/images/etiquette.png",  # Remplacez par le chemin de votre image
                                style={
                                    "display": "block",
                                    "margin": "10px auto",  # Centrer l'image horizontalement
                                    "maxWidth": "100%",  # Pour que l'image soit responsive
                                    "height": "200px"  # Pour que l'image conserve ses proportions
                                }
                            ),
                        ],
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
