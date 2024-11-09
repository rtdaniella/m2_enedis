from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

class FeatureSelectorRegression(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.selector = None

    def fit(self, X, y):
        # Initialisation du RandomForestRegressor pour la sélection des features
        model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        
        # Entraînement du modèle de régression
        model.fit(X, y)
        
        # Sélection des caractéristiques importantes
        self.selector = SelectFromModel(model)
        
        # Apprentissage du modèle et de la sélection des features
        self.selector.fit(X, y)
        
        return self

    def transform(self, X):
        # Transformation des données en utilisant la sélection des features
        return self.selector.transform(X)

    def get_support(self):
        # Retourner les indices des features sélectionnées
        return self.selector.get_support()

    def get_selected_features(self):
        # Retourner les noms des features sélectionnées
        return [col for col, mask in zip(self.feature_names_, self.get_support()) if mask]
