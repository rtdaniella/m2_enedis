import os
import sys
import pandas as pd
from .regressor_model import load_regressor  # Fonction pour charger le modèle de régression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # Ajout du chemin pour importer les modules parent

# Charger les modèles de classification et régression
regressor = load_regressor()  # Chargement du modèle de régression

# Liste des champs du formulaire (ordonnée selon les données attendues)
form_fields_regressor = [
    'classe_altitude', 'code_postal_ban',
       'cout_total_5_usages','etiquette_dpe', 'hauteur_sous_plafond', 'logement',
       'n_etage_appartement', 'passoire_energetique', 'periode_construction',
       'qualite_isolation_enveloppe', 'qualite_isolation_menuiseries',
       'qualite_isolation_murs', 'qualite_isolation_plancher_bas',
       'surface_habitable_logement', 'timestamp_reception_dpe',
       'type_batiment', 'type_energie_n_1', 'type_installation_chauffage',
       'zone_climatique'
]

# Fonction de prédiction
def predict_regressor(inputs):
    """
    Cette fonction prend en entrée un ensemble de données saisies dans le formulaire,
    applique un modèle de prédiction et retourne le résultat.

    :param inputs: Liste des données saisies par l'utilisateur dans le formulaire
    :return: Prédiction du modèle sous forme de texte
    """

    # Charger le modèle de régression
    model = load_regressor()
    if model is None:
        return "Erreur lors du chargement du modèle."

    # Récupérer les données du formulaire sous forme de dictionnaire
    input_data = {form_fields_regressor[i]: inputs[i] for i in range(len(inputs))}

    # Convertir les données du formulaire en DataFrame
    input_df = pd.DataFrame([input_data])

    # Appliquer le pipeline complet pour la prédiction
    try:
        prediction = model.predict(input_df)  # Le pipeline complet applique toutes les étapes internes
    except Exception as e:
        return f"Erreur lors du prétraitement et de la prédiction : {e}"

    # Retourner le résultat de la prédiction
    return f"Prédiction : {prediction[0]} kWhef/an"