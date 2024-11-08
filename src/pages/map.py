from dash import html

def create_map_page():
    return html.Div([
        # Bloc du titre avec fond en dégradé bleu vif à bleu foncé
        html.Div([
            html.I(className="fas fa-globe-americas",  # Icône Font Awesome
                   style={
                       "fontSize": "36px",
                       "color": "#ffffff",  # Icône blanche
                       "marginRight": "15px",
                       "verticalAlign": "middle",
                   }),
            html.H1(
                "Carte Interactives",
                style={
                    "color": "#ffffff",  # Couleur du texte en blanc
                    "fontSize": "32px",  # Taille de police plus grande
                    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",  # Typographie moderne
                    "fontWeight": "bold",
                    "display": "inline-block",  # Affichage en bloc pour centrer le texte
                    "lineHeight": "1.2",  # Un peu plus d'espace entre les lignes
                    "textAlign": "center",  # Centrer le texte
                    "marginBottom": "0",  # Pas de marge en bas pour centrer l'élément
                }
            )
        ], style={
            "background": "linear-gradient(135deg, #2a6cb2, #1a3d63)",  # Dégradé bleu plus profond et moins clair
            "padding": "20px",
            "borderRadius": "15px",  # Coins arrondis
            "boxShadow": "0 4px 10px rgba(0, 0, 0, 0.1)",  # Ombre légère pour donner de la profondeur
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",  # Centrer le contenu horizontalement
            "marginBottom": "30px",
            "marginLeft": "260px",  # Décalage pour la SideNav
            "marginTop": "20px",
            "marginRight": "10px"
        }),

        # Contenu principal de la page
        html.Div([
            html.P("Ceci est le contenu de la page Map. Explorez des cartes interactives et des données en temps réel."),
        ], style={
            "padding": "20px",
            "marginLeft": "260px",  # Décalage pour la SideNav
            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",  # Typographie moderne
            "fontSize": "16px",  # Taille du texte principale
            "color": "#333333",  # Texte sombre pour un bon contraste
        })
    ])
