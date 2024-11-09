from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from utils.models.classifier_model import load_classifier
from utils.models.regressor_model import load_regressor

# Charger les modèles
classifier = load_classifier()
regressor = load_regressor()

# Fonction pour créer la page de prédiction
def create_prediction_page():
    return html.Div([
        html.H1("Prédiction DPE et Consommation d'Énergie"),
        
        # Formulaire pour la saisie des paramètres de l'utilisateur
        html.Div([
            html.Label("Entrée 1:"),
            dcc.Input(id='input-1', type='number', placeholder='Valeur de l\'entrée 1'),
            
            html.Label("Entrée 2:"),
            dcc.Input(id='input-2', type='number', placeholder='Valeur de l\'entrée 2'),
            
            # Ajouter d'autres champs d'entrée en fonction des besoins
            
            # Bouton pour lancer la prédiction
            dbc.Button("Lancer la prédiction", id='predict-button', n_clicks=0)
        ], style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),
        
        # Afficher le résultat de la prédiction
        html.Div(id='prediction-output', style={'marginTop': '20px'})
    ])

# Callback pour effectuer la prédiction lorsque le bouton est cliqué
def register_prediction_callbacks(app):
    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [Input('input-1', 'value'), Input('input-2', 'value')]  # Ajoutez plus d'inputs si nécessaire
    )
    def update_prediction(n_clicks, input1, input2):
        if n_clicks > 0:
            # Vérifier si les entrées sont valides
            if input1 is None or input2 is None:
                return "Veuillez entrer toutes les valeurs nécessaires."

            # Préparer les données d'entrée sous forme de DataFrame
            df_input = pd.DataFrame([[input1, input2]], columns=['feature_1', 'feature_2'])  # Remplacez par les noms réels des colonnes

            # Effectuer la prédiction en utilisant le modèle choisi
            try:
                # Choisir le modèle : classifier ou regressor selon la logique de l'application
                y_pred = classifier.predict(df_input)  # Utilisez regressor.predict(df_input) si nécessaire
                return f"Résultat de la prédiction : {y_pred[0]}"
            except Exception as e:
                return f"Erreur lors de la prédiction : {str(e)}"
