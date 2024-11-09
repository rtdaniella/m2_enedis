import pandas as pd
from classifier_model import load_classifier
from regressor_model import load_regressor

# Charger les modèles
classifier = load_classifier()
regressor = load_regressor()

def predict(model_type, input_data):
    
    # Convertir les paramètres d'entrée en DataFrame
    df_input = pd.DataFrame([input_data])  # Chaque entrée devient une ligne dans le DataFrame
    
    # Effectuer la prédiction en fonction du modèle choisi
    if model_type == 'classification':
        if classifier is None:
            return "Erreur : Le modèle de classification n'a pas été chargé."
        # Utiliser le modèle de classification pour faire une prédiction
        y_pred = classifier.predict(df_input)  # Le pipeline appliquera le prétraitement automatiquement
        return f"Résultat de la classification : {y_pred[0]}"  # Afficher la classe prédite

    elif model_type == 'regression':
        if regressor is None:
            return "Erreur : Le modèle de régression n'a pas été chargé."
        # Utiliser le modèle de régression pour faire une prédiction
        y_pred = regressor.predict(df_input)  # Le pipeline appliquera le prétraitement automatiquement
        return f"Prédiction de régression : {y_pred[0]:.2f}"  # Afficher la valeur prédite

    return "Erreur : Modèle non reconnu"
