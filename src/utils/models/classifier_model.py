import logging
import os
import sys
import joblib
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline  # Import the correct pipeline
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score
from imblearn.over_sampling import BorderlineSMOTE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.models.preprocess import (
    destroy_indexes, feature_selection_classification, pca_app, preprocess_all_data,
    preprocess_pipeline, preprocess_splitted_data
)


# Configuration de base du logger pour suivre les événements de l'exécution
logging.basicConfig(
    filename='app_classifier.log',  # Fichier de log
    filemode='w',        # Mode d'écriture (écrasera le fichier à chaque exécution)
    level=logging.DEBUG, # Niveau de log (DEBUG pour plus de détails)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format du log
)


def train_classifier():
    """
    Fonction principale pour entraîner un modèle de classification avec un pipeline amélioré.
    :return: None
    """
    try:
        # Chargement des données depuis les fichiers CSV
        dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
        dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv', dtype={'column_name': str})
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        print("Erreur lors du chargement des données :", e)
        return
    
    logging.info("Démarrage du prétraitement des données.")

    # Prétraitement des données
    df, df_all = preprocess_all_data(dfa, dfn)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test, num_features, cat_features = preprocess_splitted_data(df, 'etiquette_dpe')
    X_train, X_test, y_train, y_test = destroy_indexes(X_train, X_test, y_train, y_test)

    # Vérification des colonnes manquantes
    logging.info("Vérification des données de caractéristiques.")
    missing_num_features = [col for col in num_features if col not in X_train.columns]
    missing_cat_features = [col for col in cat_features if col not in X_train.columns]
    if missing_num_features or missing_cat_features:
        logging.error(f"Colonnes manquantes : {missing_num_features + missing_cat_features}")
        return
    logging.info("les variables numériques sont : %s", num_features)
    logging.info("les variables catégorielles sont : %s", cat_features)
    logging.info("les colonnes sont :  %s", X_train.columns)
    logging.info(X_train['hauteur_sous_plafond'].head())
    # X_train.to_csv("data_classification.csv", index=False)

    # Créer et préparer le pipeline de prétraitement
    logging.info("Préparation du pipeline de prétraitement.")
    preprocessor = preprocess_pipeline(X_train, num_features, cat_features)
    
    # Appliquer le prétraitement aux données d'entraînement
    logging.info("Transformation des données avec le pipeline de prétraitement.")
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Sélection des caractéristiques les plus importantes
    logging.info("Début de la sélection des caractéristiques.")
    X_train_selected, selector = feature_selection_classification(X_train_transformed, y_train)

    logging.info(f"Forme des données après sélection des caractéristiques : {X_train_selected.shape}")

    # Déterminer le nombre optimal de composants PCA
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
    
    param_dist  = {
        'pca__n_components': [optimal_components,5,2],
        'classifier__n_estimators': [50,200],
        'classifier__max_depth': [None, 30,50]
    }

    randomized_search = RandomizedSearchCV(
        estimator=pipeline,  # Utiliser le pipeline complet 
        param_distributions=param_dist,
        n_iter=20,  # Nombre d'itérations pour RandomizedSearchCV
        cv=StratifiedKFold(n_splits=4),  # Validation croisée
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Entraîner le modèle avec RandomizedSearchCV sur le pipeline complet
    randomized_search.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = randomized_search.predict(X_test)

    # Rapport de classification
    logging.info("Rapport de classification :")
    logging.info(classification_report(y_test, y_pred))
    logging.info("Accuracy : %f", accuracy_score(y_test, y_pred))
    logging.info("Recall : %f", recall_score(y_test, y_pred, average='macro'))
    logging.info("Precision : %f", precision_score(y_test, y_pred, average='macro'))
    
    # Sauvegarde du meilleur modèle avec versionnage
    model_version = 1
    model_path = f'models/classifier_model_v{model_version}.joblib'
    
    # Vérifier la version existante du modèle et incrémenter
    while os.path.exists(model_path):
        model_version += 1
        model_path = f'src/utils/models/classifier_model_v{model_version}.joblib'
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(randomized_search.best_estimator_, model_path)  # Sauvegarder le modèle avec le pipeline optimal
    logging.info(f"Modèle sauvegardé sous {model_path}")

def load_classifier():
    """
    Fonction pour charger un modèle sauvegardé avec gestion de version.
    :return: Le modèle chargé ou None si le modèle n'existe pas.
    """
    model_version = 1
    model_path = f'src/utils/models/classifier_model_v{model_version}.joblib'

    while os.path.exists(model_path):
        model_version += 1
        model_path = f'src/utils/models/classifier_model_v{model_version}.joblib'

    model_path = f'src/utils/models/classifier_model_v{model_version - 1}.joblib'

    if not os.path.exists(model_path):
        logging.error("Erreur : le modèle n'a pas été trouvé.")
        print("Erreur : le modèle n'a pas été trouvé.")
        return None

    return joblib.load(model_path)