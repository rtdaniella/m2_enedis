import logging
import os
import sys
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.models.feature_selector import FeatureSelectorRegression
from utils.models.preprocess import destroy_indexes, pca_app, preprocess_all_data, preprocess_pipeline, preprocess_splitted_data

def train_regressor():
    # Import des données DPE sous forme de dataframes
    try:
        dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
        dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv')
    except Exception as e:
        print("Erreur lors du chargement des données :", e)
        return
    logging.info("Démarrage du prétraitement des données.")
    
    # Prétraitement des données
    df, df_all = preprocess_all_data(dfa, dfn)
    X_train, X_test, y_train, y_test, num_features, cat_features = preprocess_splitted_data(df, 'conso_5_usages_e_finale',False)
    X_train, X_test, y_train, y_test = destroy_indexes(X_train, X_test, y_train, y_test)
    
    # Gérer les valeurs manquantes
    # X_test, y_test = handle_missing_values_in_X_y_test(X_test, y_test, num_features, cat_features)

    # Vérification des colonnes manquantes
    logging.info("Vérification des données de caractéristiques.")
    missing_num_features = [col for col in num_features if col not in X_train.columns]
    missing_cat_features = [col for col in cat_features if col not in X_train.columns]
    if missing_num_features or missing_cat_features:
        logging.error(f"Colonnes manquantes : {missing_num_features + missing_cat_features}")
        return
    
    # Définir le pipeline de prétraitement
    logging.info("Préparation du pipeline de prétraitement.")
    preprocessor = preprocess_pipeline(X_train, num_features, cat_features)
    feature_selector = FeatureSelectorRegression()  # Assurer que cette classe est bien définie dans ton code

    # # Transformation des données d'entraînement
    # logging.info("Transformation des données avec le pipeline de prétraitement.")
    # X_train_transformed = preprocessor.fit_transform(X_train)

    # # Sélectionner les features les plus importantes
    # logging.info("Début de la sélection des caractéristiques.")
    # feature_selector = FeatureSelectorRegression()  # Classe que nous avons définie
    # feature_selector.fit(X_train_transformed, y_train)
    
    # # Appliquer la sélection des caractéristiques
    # X_train_selected = feature_selector.transform(X_train_transformed)
    # logging.info(f"Forme des données après sélection des caractéristiques : {X_train_selected.shape}")

    # # Transformation des données de test
    # X_test_transformed = preprocessor.transform(X_test)
    # X_test_selected = feature_selector.transform(X_test_transformed)
    # logging.info(f"Forme des données de test après sélection des caractéristiques : {X_test_selected.shape}")

    # # Vérification de la correspondance des dimensions
    # if X_train_selected.shape[1] != X_test_selected.shape[1]:
    #     logging.error("Les dimensions de X_train_selected et X_test_selected ne correspondent pas.")
    #     return

    # # Déterminer le nombre optimal de composants avec PCA après sélection des features
    # logging.info("Détermination du nombre optimal de composants PCA.")
    # optimal_components = pca_app(pd.DataFrame(X_test_selected))  # PCA sur X_train pour déterminer le nombre optimal de composants
    
    # # Appliquer PCA sur X_train_selected
    # pca = PCA(svd_solver='full', n_components=optimal_components)
    # X_train_pca = pca.fit_transform(X_train_selected)  # Ajustement de PCA et transformation de X_train_selected
    
    # # Appliquer la même transformation PCA sur X_test_selected
    # X_test_pca = pca.transform(X_test_selected)  # Transforme X_test_selected en utilisant les mêmes composants
    
    # logging.info(f"Forme des données d'entraînement après PCA : {X_train_pca.shape}")
    # logging.info(f"Forme des données de test après PCA : {X_test_pca.shape}")

    # Pipeline de modélisation avec PCA et RandomForestRegressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        # ('pca', PCA(svd_solver='full')), 
        ('regressor', RandomForestRegressor())
    ])
    
     # Déterminer le nombre optimal de composants avec PCA après sélection des features
    # logging.info("Détermination du nombre optimal de composants PCA.")
    # optimal_components = pca_app(pd.DataFrame(X_train))  # PCA sur X_train pour déterminer le nombre optimal de composants

    # Grille de paramètres pour GridSearchCV
    param_grid = {
        'regressor__n_estimators': [200],
        'regressor__max_depth': [30],
        #  'pca__n_components': [4],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Utilisation de l'erreur quadratique moyenne pour la régression
        cv=KFold(n_splits=2),
        n_jobs=-1,
        verbose=1
    )
    
    # Fit avec les données d'entraînement (X_train_pca)
    grid_search.fit(X_train, y_train)

    # Meilleurs paramètres
    print("Meilleurs paramètres trouvés :", grid_search.best_params_)

    # Évaluation du modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)  # Utilise X_test_pca après PCA

    # Métriques de régression
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error (MAE) sur l'ensemble de test : {mean_absolute_error(y_test, y_pred)} kWh/an:" )
    print("R2 Score:", r2_score(y_test, y_pred))

def load_regressor():
    # Charger le modèle
    model_path = 'models/regressor_model.joblib'
    if not os.path.exists(model_path):
        print("Erreur : le modèle n'a pas été trouvé.")
        return None
    return joblib.load(model_path)
