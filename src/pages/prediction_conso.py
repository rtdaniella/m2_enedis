import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

# Charger le fichier CSV des données DPE
df = pd.read_csv("src/files/data_regressor.csv")

# Liste des variables à inclure dans le formulaire
form_fields_conso = [
    'classe_altitude', 'code_postal_ban',
       'cout_total_5_usages','etiquette_dpe', 'hauteur_sous_plafond', 'logement',
       'n_etage_appartement', 'passoire_energetique', 'periode_construction',
       'qualite_isolation_enveloppe', 'qualite_isolation_menuiseries',
       'qualite_isolation_murs', 'qualite_isolation_plancher_bas',
       'surface_habitable_logement', 'timestamp_reception_dpe',
       'type_batiment', 'type_energie_n_1', 'type_installation_chauffage',
       'zone_climatique'
]

cat_features__regressor =['classe_altitude', 'code_postal_ban','etiquette_dpe', 'logement', 'n_etage_appartement',
       'passoire_energetique', 'periode_construction',
       'qualite_isolation_enveloppe', 'qualite_isolation_menuiseries',
       'qualite_isolation_murs', 'qualite_isolation_plancher_bas',
       'type_batiment', 'type_energie_n_1', 'type_installation_chauffage',
       'zone_climatique'] 
num_features__regressor = ['cout_total_5_usages',
       'hauteur_sous_plafond', 'surface_habitable_logement',
       'timestamp_reception_dpe'] 

# Fonction pour générer les champs de formulaire dynamiquement
def generate_form_fields(df, form_fields, cat_features, num_features):
    form_inputs = []
    
    for column in form_fields:
        if column in df.columns:
            if column in cat_features:
                unique_values = df[column].dropna().unique()
                options = [{"label": str(val), "value": str(val)} for val in unique_values]
                field = dbc.Row([
                    dbc.Label(column.replace("_", " ").capitalize(), width=4),
                    dbc.Col(dcc.Dropdown(
                        id=f"{column}-input", options=options,
                        placeholder=f"Sélectionnez {column.replace('_', ' ')}", clearable=True), width=8
                    )
                ], className="mb-3")
            elif column in num_features:
                field = dbc.Row([
                    dbc.Label(column.replace("_", " ").capitalize(), width=4),
                    dbc.Col(dcc.Input(
                        id=f"{column}-input", type="number",
                        placeholder=f"Entrez {column.replace('_', ' ')}"), width=8
                    )
                ], className="mb-3")
            else:
                field = dbc.Row([
                    dbc.Label(column.replace("_", " ").capitalize(), width=4),
                    dbc.Col(dcc.Input(
                        id=f"{column}-input", type="text",
                        placeholder=f"Entrez {column.replace('_', ' ')}"), width=8
                    )
                ], className="mb-3")
            form_inputs.append(field)
        else:
            print(f"Avertissement : '{column}' n'est pas présent dans les colonnes du DataFrame.")
    
    return form_inputs

form_inputs = generate_form_fields(df, form_fields_conso, cat_features__regressor, num_features__regressor)

# Fonction de création de la page avec le formulaire
def create_pred_conso_page():
    layout = html.Div(
        [   

            # Bloc du titre avec fond en dégradé bleu vif à bleu foncé
            html.Div(
                [
                    html.I(
                        className="fas fa-cogs",  # Icône Font Awesome
                        style={
                            "fontSize": "36px",
                            "color": "#ffffff",  # Icône blanche
                            "marginRight": "15px",
                            "verticalAlign": "middle",
                        },
                    ),
                    html.H1(
                        "Prédiction de la consommation énergétique",
                        style={
                            "color": "#ffffff",  # Couleur du texte en blanc
                            "fontSize": "32px",  # Taille de police plus grande
                            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",  # Typographie moderne
                            "fontWeight": "bold",
                            "display": "inline-block",  # Affichage en bloc pour centrer le texte
                            "lineHeight": "1.2",  # Un peu plus d'espace entre les lignes
                            "textAlign": "center",  # Centrer le texte
                            "marginBottom": "0",  # Pas de marge en bas pour centrer l'élément
                        },
                    ),
                ],
                style={
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
                    "marginRight": "10px",
                    "width":"1200px"
                },
            ),

            # Bloc principal avec deux colonnes
            dbc.Row(
                [
                    # Colonne de formulaire
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Informations du logement",
                                        className="card-title",
                                    ),
                                    *form_inputs,  # Insertion des champs de formulaire générés dynamiquement
                                    dbc.Button(
                                        "Prédire",
                                        id="predict-button-conso",                                     
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style={
                                "padding": "20px",
                                "margin-right": "15px",
                                "backgroundColor": "lightgrey",
                            },
                        ),
                        width=6,
                    ),
                    # Colonne pour le résultat de la prédiction
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Estimation de votre consommation énergétique",
                                        className="card-title",
                                        style={
                                            "fontSize": "24px",  # Taille de police du titre
                                            "fontWeight": "bold",  # Police en gras
                                            "color": "#333",  # Couleur sombre pour le titre
                                        },
                                    ),
                                    html.Div(
                                        id="prediction-result-conso",
                                        className="mt-3",
                                        style={
                                            "fontSize": "36px",  # Taille de police du résultat
                                            "fontWeight": "bold",  # Police en gras
                                            "color": "#ffffff",  # Texte blanc pour contraster avec l'arrière-plan
                                            "backgroundColor": "#34c5a8",
                                            "padding": "20px",  # Espacement autour du texte
                                            "borderRadius": "10px",  # Bordures arrondies pour donner un effet doux
                                            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.2)",  # Ombre pour donner un effet de profondeur
                                            "textAlign": "center",  # Centrer le texte horizontalement
                                            "verticalAlign": "middle",  # Centrer le texte verticalement
                                            "minHeight": "150px",  # Hauteur minimale du bloc pour garantir que le texte est bien centré
                                            "display": "flex",  # Utiliser flexbox pour aligner le texte
                                            "justifyContent": "center",  # Centrer le texte sur l'axe horizontal
                                            "alignItems": "center",  # Centrer le texte sur l'axe vertical
                                            "transition": "all 0.3s ease-in-out",  # Transition douce pour les effets
                                        },
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f0f0f0",  # Fond gris clair pour le bloc
                                "borderRadius": "10px",  # Coins arrondis pour le bloc
                                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",  # Ombre légère autour du bloc
                                "marginTop":"150px"
                            },
                        ),
                        width=6,
                    ),
                ],
                justify="center",
                style={"padding": "20px", "marginLeft": "260px"},
            ),  # Ajustement du positionnement
        ]
    )

    return layout

