import os
import sys
import pandas as pd
from .classifier_model import load_classifier  # Fonction pour charger le modèle de classification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # Ajout du chemin pour importer les modules parent

# Charger les modèles de classification et régression
classifier = load_classifier()  # Chargement du modèle de classification

# Liste des champs du formulaire 
form_fields_classifier = [
    'classe_altitude', 'code_postal_ban', 'conso_5_usages_e_finale',
    'cout_total_5_usages', 'hauteur_sous_plafond', 'logement',
    'n_etage_appartement', 'passoire_energetique', 'periode_construction',
    'qualite_isolation_enveloppe', 'qualite_isolation_menuiseries',
    'qualite_isolation_murs', 'qualite_isolation_plancher_bas',
    'surface_habitable_logement', 'timestamp_reception_dpe',
    'type_batiment', 'type_energie_n_1', 'type_installation_chauffage',
    'zone_climatique'
]

# Fonction de prédiction
def predict_classifier(inputs):
    """
    Cette fonction prend en entrée un ensemble de données saisies dans le formulaire,
    applique un modèle de prédiction et retourne le résultat.

    :param inputs: Liste des données saisies par l'utilisateur dans le formulaire
    :return: Prédiction du modèle sous forme de texte
    """

    # Étape 1 : Charger le modèle de classification
    model = load_classifier()  
    
    # Vérification que le modèle est bien chargé
    if model is None:
        return "Erreur lors du chargement du modèle."  # Si le modèle ne peut pas être chargé, on renvoie une erreur.

    # Étape 2 : Préparer les données d'entrée
    # Les inputs arrivent sous forme de liste, et nous devons les convertir en dictionnaire
    input_data = {form_fields_classifier[i]: inputs[i] for i in range(len(inputs))}  # Création d'un dictionnaire clé/valeur

    # Étape 3 : Convertir les données d'entrée en DataFrame
    # La conversion en DataFrame permet de manipuler les données avec le modèle de manière cohérente
    input_df = pd.DataFrame([input_data])  # Ici on crée un DataFrame avec une ligne de données provenant du formulaire

    # Étape 4 : Faire la prédiction avec le modèle
    # On utilise le modèle chargé pour effectuer la prédiction sur les données d'entrée
    prediction = model.predict(input_df)  # Le modèle prédit en utilisant les données d'entrée sous forme de DataFrame

    # Étape 5 : Retourner le résultat de la prédiction
    # La prédiction est retournée sous forme d'une chaîne de caractères, avec le premier résultat du tableau de prédiction
    return f"Prédiction : classe {prediction[0]}"  # On retourne la prédiction obtenue
