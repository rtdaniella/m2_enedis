import logging
import os
import sys
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline  # Import the correct pipeline
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score
from imblearn.over_sampling import BorderlineSMOTE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.models.preprocess import (
    destroy_indexes, feature_selection_classification, pca_app, preprocess_all_data,
    preprocess_pipeline, preprocess_splitted_data
)


import logging

# Configuration de base du logger
logging.basicConfig(
    filename='app.log',  # Nom du fichier où enregistrer les logs
    filemode='w',        # Mode écriture (écrasera le fichier à chaque exécution)
    level=logging.DEBUG, # Niveau de logs (DEBUG est le plus détaillé)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def train_classifier():
    # Chargement des données et prétraitement comme dans votre code initial
    try:
        dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
        dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv', dtype={'column_name': str})
    except Exception as e:
        print("Erreur lors du chargement des données :", e)
        return
    
    logging.info("Démarrage du prétraitement des données.")

    # Prétraitement des données
    df, df_all = preprocess_all_data(dfa, dfn)
    X_train, X_test, y_train, y_test, num_features, cat_features = preprocess_splitted_data(df, 'etiquette_dpe')
    X_train, X_test, y_train, y_test = destroy_indexes(X_train, X_test, y_train, y_test)

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
    
    # Transformer X_train pour l'utiliser dans la sélection de features
    logging.info("Transformation des données avec le pipeline de prétraitement.")
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Sélectionner les features les plus importantes
    logging.info("Début de la sélection des caractéristiques.")
    X_train_selected, selector = feature_selection_classification(X_train_transformed, y_train)

    logging.info(f"Forme des données après sélection des caractéristiques : {X_train_selected.shape}")

    # Déterminer le nombre optimal de composantes avec PCA après sélection des features
    logging.info("Détermination du nombre optimal de composants PCA.")
    optimal_components = pca_app(pd.DataFrame(X_train_selected))

    # Pipeline de modélisation avec SMOTE, sélection de features, PCA et RandomForest
    logging.info("Création du pipeline de modélisation.")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', BorderlineSMOTE(random_state=42)),
        ('feature_selection', selector),  # Sélection des features
        ('pca', PCA(svd_solver='full', n_components=optimal_components)),
        ('classifier', RandomForestClassifier())
    ])

    # Grille de paramètres pour GridSearchCV
    logging.info("Démarrage de la recherche sur grille pour les meilleurs hyperparamètres.")
    param_grid = {
        'pca__n_components': [optimal_components,5,2],
        'classifier__n_estimators': [50,200],
        'classifier__max_depth': [None, 30,50]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=StratifiedKFold(n_splits=4),
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Évaluation du modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Rapport de classification
    logging.info(classification_report(y_test, y_pred))
    logging.info("Accuracy : %f", accuracy_score(y_test, y_pred))

    # Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/classifier_model.joblib')
    logging.info("Modèle sauvegardé sous 'models/classifier_model.joblib'")

def load_classifier():
    # Charger le modèle
    model_path = 'models/classifier_model.joblib'
    if not os.path.exists(model_path):
        print("Erreur : le modèle n'a pas été trouvé.")
        return None
    return joblib.load(model_path)
