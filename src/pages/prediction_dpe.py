import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc

# Charger le fichier CSV
df = pd.read_csv(
    "src/files/dpe-nettoye.csv"
)  # Remplace "ton_fichier.csv" par le chemin réel de ton fichier CSV

# Liste des variables à inclure dans le formulaire
form_fields = [
    "type_batiment",
    "code_postal_ban",
    "classe_altitude",
    "cout_total_5_usages",
    "etiquette_dpe",
    "qualite_isolation_menuiseries",
    "type_installation_chauffage",
    "conso_5_usages_e_finale",
    "qualite_isolation_enveloppe",
    "qualite_isolation_plancher_bas",
    "zone_climatique",
    "n_dpe",
    "geopoint",
    "surface_habitable_logement",
    "logement",
    "qualite_isolation_murs",
    "n_etage_appartement",
    "type_energie_n_1",
    "hauteur_sous_plafond",
    "passoire_energetique",
    "periode_construction",
    "timestamp_reception_dpe",
]


# Générer dynamiquement les champs en fonction des valeurs uniques dans le CSV, uniquement pour les champs spécifiés
def generate_form_fields(df, fields):
    form_fields = []

    for column in fields:
        # Vérifier si la colonne existe dans le DataFrame pour éviter les erreurs
        if column in df.columns:
            # Récupérer les valeurs uniques de la colonne
            unique_values = df[column].dropna().unique()

            # Si la colonne a un nombre limité de valeurs uniques, on utilise une liste déroulante
            if len(unique_values) <= 20:  # Par exemple, si moins de 20 valeurs uniques
                options = [
                    {"label": str(val), "value": str(val)} for val in unique_values
                ]
                field = dbc.Row(
                    [
                        dbc.Label(column.replace("_", " ").capitalize(), width=4),
                        dbc.Col(
                            dcc.Dropdown(
                                id=f"{column}-input",
                                options=options,
                                placeholder=f"Sélectionnez {column.replace('_', ' ')}",
                                clearable=True,
                            ),
                            width=8,
                        ),
                    ],
                    className="mb-3",
                )

            # Sinon, on utilise un champ texte
            else:
                field = dbc.Row(
                    [
                        dbc.Label(column.replace("_", " ").capitalize(), width=4),
                        dbc.Col(
                            dbc.Input(
                                type="text",
                                id=f"{column}-input",
                                placeholder=f"Entrez {column.replace('_', ' ')}",
                            ),
                            width=8,
                        ),
                    ],
                    className="mb-3",
                )

            form_fields.append(field)

    return form_fields


# Générer les champs de formulaire à partir du CSV pour les variables spécifiées
form_inputs = generate_form_fields(df, form_fields)


# Fonction de création de la page avec le formulaire
def create_pred_dpe_page():
    return html.Div(
        [
            html.H1(
                "Prediction DPE",
                style={"text-align": "center", "margin-bottom": "20px"},
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
                                        "Formulaire de prédiction",
                                        className="card-title",
                                    ),
                                    *form_inputs,  # Insertion des champs de formulaire générés dynamiquement
                                    dbc.Button(
                                        "Prédire",
                                        id="predict-button",
                                        color="primary",
                                        className="mt-3",
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
                                        "Résultat de la prédiction",
                                        className="card-title",
                                    ),
                                    html.Div(
                                        id="prediction-result", className="mt-3"
                                    ),  # Zone d'affichage du résultat
                                ]
                            ),
                            className="shadow-sm",
                            style={"padding": "20px", "backgroundColor": "lightgrey"},
                        ),
                        width=6,
                    ),
                ],
                justify="center",
                style={"margin-top": "20px", "margin-left": "260px"},
            ),  # Ajustement du positionnement
        ]
    )
