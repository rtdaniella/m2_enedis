from dash import html

def create_not_found_page():
    return html.Div([
        # Image WebP pour la page Not Found
        html.Div(
            children=[
                html.Img(
                    src="/assets/gif/notfound.webp",  # Le chemin d'accès au fichier WebP dans le dossier assets
                     style={
                        "width": "50%",   # Taille de l'image (ajustée à 30% de la largeur de la page)
                        "height": "auto", # Hauteur automatique pour garder les proportions
                        # "position": "absolute",  # Pour positionner l'image à droite
                        "right": "10px",   # Position à 10px du bord droit de la page
                        "top": "20px",     # Position à 20px du bord supérieur de la page
                    }
                )
            ],
            style={"textAlign": "center", "marginTop": "50px"}  # Style pour centrer l'image
        )
    ])
