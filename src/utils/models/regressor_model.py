import logging
import os
import sys
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.models.feature_selector import FeatureSelectorRegression
from utils.models.preprocess import destroy_indexes, preprocess_all_data, preprocess_pipeline, preprocess_splitted_data

# Configuration du logger pour enregistrer les événements et erreurs
logging.basicConfig(
    filename='app_regression.log',  # Nom du fichier de log
    filemode='w',        # Mode d'écriture du fichier de log (écrasera à chaque exécution)
    level=logging.DEBUG, # Niveau des logs (DEBUG permet d'avoir tous les détails)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format du log
)

def train_regressor():
    """
    Fonction d'entraînement du modèle de régression.
    Cette fonction charge les données, les prépare, applique la sélection de caractéristiques,
    et entraîne un modèle de régression avec RandomizedSearchCV pour optimiser les hyperparamètres.
    Enfin, elle évalue le modèle et le sauvegarde dans un fichier.
    """
    
    # Import des données DPE sous forme de dataframes depuis les URLs
    try:
        dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
        dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv')
    except Exception as e:
        print("Erreur lors du chargement des données :", e)
        return
    
    # Log du début du prétraitement des données
    logging.info("Démarrage du prétraitement des données.")
    
    # Prétraitement des données (nettoyage, transformation, etc.)
    df, df_all = preprocess_all_data(dfa, dfn)
    # Préparation des données d'entraînement et de test
    X_train, X_test, y_train, y_test, num_features, cat_features = preprocess_splitted_data(df, 'conso_5_usages_e_finale', False)
    X_train, X_test, y_train, y_test = destroy_indexes(X_train, X_test, y_train, y_test)
    
    # Vérification de la présence des colonnes nécessaires dans les jeux de données
    logging.info("Vérification des données de caractéristiques.")
    missing_num_features = [col for col in num_features if col not in X_train.columns]
    missing_cat_features = [col for col in cat_features if col not in X_train.columns]
    if missing_num_features or missing_cat_features:
        logging.error(f"Colonnes manquantes : {missing_num_features + missing_cat_features}")
        return

    logging.info("les variables numériques sont : %s", num_features)
    logging.info("les variables catégorielles sont : %s", cat_features)
    logging.info("les colonnes sont :  %s", X_train.columns)
    # X_train.to_csv("data_regressor.csv", index=False)

    # Préparation du pipeline de prétraitement avec normalisation et imputation
    logging.info("Préparation du pipeline de prétraitement.")
    preprocessor = preprocess_pipeline(X_train, num_features, cat_features)
    
    # Sélection des caractéristiques importantes
    feature_selector = FeatureSelectorRegression()  # Classe définie pour la sélection de features
    
   
    # Pipeline de modélisation avec PCA et RandomForestRegressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('pca', PCA(svd_solver='full')), 
        ('regressor', RandomForestRegressor())
    ])
    
      # Log de l'étape de recherche des hyperparamètres
    logging.info("Recherche des meilleurs hyperparamètres avec RandomizedSearchCV.")
    
    # Grille de paramètres pour RandomizedSearchCV


    param_grid = {
        'regressor__n_estimators':[100, 200, 300],
        'regressor__max_depth': [20, 30, None],
        'pca__n_components': [3,5,8],
        'regressor__min_samples_split': [2, 5, 10],                  # Nombre minimum d'échantillons pour diviser un noeud
        'regressor__min_samples_leaf': [1, 2, 4],                    # Nombre minimum d'échantillons dans chaque feuille
        'regressor__max_features': ['auto', 'sqrt', 'log2']         # Nombre de caractéristiques à considérer lors de la division
    }

     # Application de RandomizedSearchCV avec un échantillonnage aléatoire des hyperparamètres
    random_search = RandomizedSearchCV(
        estimator=pipeline,                              # Estimator à ajuster
        param_distributions=param_grid,                  # Grille des paramètres à tester
        n_iter=20,                                       # Nombre d'itérations pour explorer le paramètre
        scoring='neg_mean_squared_error',                # Mesure de performance (négatif de l'erreur quadratique moyenne)
        cv=KFold(n_splits=5),                            # Validation croisée avec 5 plis
        n_jobs=-1,                                       # Utilisation de tous les cœurs de CPU
        verbose=1,                                       # Affichage de la progression
        random_state=42                                  # Fixation du random_state pour la reproductibilité
    )

    # Entraînement du modèle avec les données d'entraînement
    random_search.fit(X_train, y_train)

    # Log des meilleurs paramètres trouvés par RandomizedSearchCV
    logging.info(f"Meilleurs paramètres trouvés : {random_search.best_params_}")

    # Modèle final après recherche des meilleurs hyperparamètres
    best_model = random_search.best_estimator_
    
    # Prédictions sur les données de test
    y_pred = best_model.predict(X_test)

    # Calcul des métriques de régression pour évaluer la performance du modèle
    mse = mean_squared_error(y_test, y_pred)        # Erreur quadratique moyenne (MSE)
    mae = mean_absolute_error(y_test, y_pred)        # Erreur absolue moyenne (MAE)
    r2 = r2_score(y_test, y_pred)                    # Score R2, indicatif de la qualité de la prédiction

    # Log des résultats de performance du modèle
    logging.info(f"Mean Squared Error (MSE) : {mse}")
    logging.info(f"Mean Absolute Error (MAE) : {mae}")
    logging.info(f"R2 Score : {r2}")

     # Sauvegarde du meilleur modèle avec versionnage
    model_version = 1
    model_path = f'models/regressor_model_v{model_version}.joblib'
    
    # Vérifier la version existante du modèle et incrémenter
    while os.path.exists(model_path):
        model_version += 1
        model_path = f'src/utils/models/regressor_model_v{model_version}.joblib'
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(random_search.best_estimator_, model_path, compress=3)  # Sauvegarder le modèle avec le pipeline optimal
    logging.info(f"Modèle sauvegardé sous {model_path}")


def load_regressor():
    """
    Charge le modèle complet de régression (pipeline) sauvegardé.
    """
    model_version = 1
    model_path = f'src/utils/models/regressor_model_v{model_version}.joblib'

    while os.path.exists(model_path):
        model_version += 1
        model_path = f'src/utils/models/regressor_model_v{model_version}.joblib'

    model_path = f'src/utils/models/regressor_model_v{model_version - 1}.joblib'

    if not os.path.exists(model_path):
        print("Erreur : le modèle n'a pas été trouvé.")
        return None

    return joblib.load(model_path , mmap_mode='r')