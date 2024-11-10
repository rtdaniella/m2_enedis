from dash import html

def create_about_page():
    return html.Div([
        # Titre principal
        html.H1(
            "À propos de l'application", 
            style={
                "textAlign": "center", 
                "color": "#ffffff", 
                "marginTop": "40px", 
                "fontFamily": "Arial, sans-serif", 
                "fontSize": "36px", 
                "fontWeight": "bold",
                "textShadow": "2px 2px 5px rgba(0, 0, 0, 0.3)"
            }
        ),
        
        # Bloc principal avec des infos centrées et stylées
        html.Div([
            html.Div([

                html.P(
                    "Cette application permet d'analyser l'impact énergétique des bâtiments en fonction de leur classe de DPE "
                    "et aide les utilisateurs à optimiser leur consommation d'énergie.",
                    style={
                        "fontSize": "18px", 
                        "lineHeight": "1.6", 
                        "color": "#d1e3f3", 
                        "textAlign": "center", 
                        "marginBottom": "20px"
                    }
                ),
                
                # Contexte et objectif de l'application
                html.H3("Contexte et objectif de l'application", 
                        style={
                            "color": "#ffffff", 
                            "fontSize": "22px", 
                            "marginBottom": "10px", 
                            "fontWeight": "bold"
                        }
                ),
                html.P(
                    "Avec la montée des préoccupations environnementales et énergétiques, cette application a été conçue pour sensibiliser "
                    "les utilisateurs aux économies d'énergie en fonction de la performance énergétique des bâtiments. "
                    "L'objectif est de fournir un outil d'analyse interactif pour mieux comprendre les consommations électriques "
                    "des logements en fonction de leur classe DPE et des paramètres environnementaux.",
                    style={
                        "fontSize": "16px", 
                        "lineHeight": "1.6", 
                        "color": "#b0c4de", 
                        "textAlign": "justify", 
                        "marginBottom": "20px"
                    }
                ),
            
                
                # Équipe de développement
                html.H3("Membres du projet - Master SISE 2024-2025", 
                        style={
                            "color": "#ffffff", 
                            "fontSize": "22px", 
                            "marginBottom": "10px", 
                            "fontWeight": "bold"
                        }
                ),
                html.Ul([
                    html.Li("Daniella Rakotondratsimba"),
                    html.Li("Béranger Thomas"),
                    html.Li("Akrem Jomaa"),
                ], style={
                    "fontSize": "20px", 
                    "color": "#d1e3f3", 
                    "lineHeight": "1.6"
                })
            ], style={
                'background': 'linear-gradient(135deg, #000428, #004e92)',
                "padding": "30px",
                "borderRadius": "8px",
                "boxShadow": "0px 8px 16px rgba(0, 0, 0, 0.2)",
                "width": "70%",
                "margin": "40px auto",
                "border": "2px solid #34495e"  # Bordure subtile
            }),
        ]),

        # Style global de la page
    ], style={
        "padding": "20px",
        "marginLeft": "260px",  # Décalage pour laisser de l'espace à la SideNav
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#ecf0f1"  # Fond clair général
    })
