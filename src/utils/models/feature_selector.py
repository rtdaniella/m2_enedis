import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np

class FeatureSelectorRegression(BaseEstimator, TransformerMixin):
    """
    Classe pour la sélection des caractéristiques les plus importantes à l'aide d'un modèle RandomForestRegressor.
    Cette classe sélectionne les caractéristiques en fonction de l'importance de chaque feature donnée par un modèle
    RandomForest entraîné.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialise le sélecteur de caractéristiques avec les hyperparamètres du modèle RandomForest.
        
        :param n_estimators: Nombre d'arbres dans le modèle RandomForest (par défaut 100).
        :param random_state: Initialisation du générateur de nombres aléatoires pour la reproductibilité (par défaut 42).
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.selector = None
        self.feature_names_ = None

    def fit(self, X, y):
        """
        Entraîne un modèle RandomForestRegressor pour déterminer l'importance des caractéristiques et sélectionne
        les caractéristiques les plus pertinentes.
        
        :param X: DataFrame ou array contenant les données d'entrée.
        :param y: Vector cible (les valeurs que nous voulons prédire).
        :return: self (objet ajusté).
        """
        # Vérification si X est un DataFrame et récupération des noms de features
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            self.feature_names_ = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Initialisation du modèle RandomForestRegressor
        model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        
        # Entraînement du modèle
        model.fit(X, y)
        
        # Sélection des features les plus importantes
        self.selector = SelectFromModel(model)
        
        # Ajustement de la sélection des features sur les données
        self.selector.fit(X, y)
        
        return self

    def transform(self, X):
        """
        Applique la sélection des caractéristiques sur un nouvel ensemble de données.
        
        :param X: Données à transformer.
        :return: X transformé avec uniquement les caractéristiques sélectionnées.
        """
        # Vérification si X a le bon format
        if X.shape[1] != len(self.feature_names_):
            raise ValueError("Le nombre de colonnes dans X ne correspond pas aux features du modèle appris.")
        
        # Transformation des données en fonction de la sélection des features
        return self.selector.transform(X)

    def get_support(self):
        """
        Récupère les indices des features sélectionnées.
        
        :return: Tableau booléen indiquant quelles features ont été sélectionnées.
        """
        if self.selector is None:
            raise RuntimeError("La méthode 'fit' doit être appelée avant 'get_support'.")
        
        return self.selector.get_support()

    def get_selected_features(self):
        """
        Récupère les noms des features sélectionnées après le fit.
        
        :return: Liste des noms des features sélectionnées.
        """
        # Vérification si 'fit' a été appelé pour s'assurer que le modèle a été ajusté
        if self.selector is None:
            raise RuntimeError("La méthode 'fit' doit être appelée avant 'get_selected_features'.")
        
        # Renvoi des noms des features sélectionnées
        selected_features = np.array(self.feature_names_)[self.get_support()]
        return selected_features.tolist()
