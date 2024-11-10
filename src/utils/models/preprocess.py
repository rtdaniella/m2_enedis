import logging
from venv import logger
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold


# Module de préparation des données pour la prédiction de la consommation et de l'étiquette DPE.

import pandas as pd
import logging

# --- Fonctions de Prétraitement ---

def categoriser_annee_construction(valeur):
    """
    Catégorise l'année de construction en tranches temporelles.
    
    Paramètre:
        valeur (int): Année de construction du logement.
        
    Retourne:
        str: Catégorie temporelle de construction.
    """
    if valeur <= 1960:
        return 'Avant 1960'
    elif 1960 < valeur <= 1970:
        return '1961 - 1970'
    elif 1970 < valeur <= 1980:
        return '1971 - 1980'
    elif 1980 < valeur <= 1990:
        return '1981 - 1990'
    elif 1990 < valeur <= 2000:
        return '1991 - 2000'
    elif 2000 < valeur <= 2010:
        return '2001 - 2010'
    elif 2010 < valeur <= 2020:
        return '2011 - 2020'
    elif 2020 < valeur:
        return 'Après 2020'
    else:
        return 'Inconnue'


# Dictionnaire pour la conversion des accents en lettres sans accent
accents = {
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'à': 'a', 'â': 'a', 'ä': 'a',
    'î': 'i', 'ï': 'i',
    'ô': 'o', 'ö': 'o',
    'ù': 'u', 'û': 'u', 'ü': 'u',
    'ç': 'c'
}

# Liste des caractères spéciaux à remplacer par un underscore
special_chars = [' ', '/', '(', ')', '-', ',', '.', '°']

def clean_text(text):
    """
    Nettoie une chaîne de caractères en remplaçant les accents et caractères spéciaux.
    
    Paramètre:
        text (str): Texte à nettoyer.
        
    Retourne:
        str: Texte nettoyé, avec accents et caractères spéciaux remplacés.
    """
    # Remplace les accents par leur équivalent sans accent
    for accent, sans_accent in accents.items():
        text = text.replace(accent, sans_accent)

    # Remplace les caractères spéciaux par un underscore
    for char in special_chars:
        text = text.replace(char, '_')

    # Supprime les underscores consécutifs et en début/fin
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_').lower()


def delimiter_outliers(df, col, factor=1.5):
    """
    Calcule les bornes pour identifier les valeurs aberrantes en utilisant l'IQR (Intervalle Interquartile).
    
    Paramètres:
        df (DataFrame): Données contenant la colonne à analyser.
        col (str): Nom de la colonne cible.
        factor (float): Facteur d'étendue pour définir les limites, par défaut 1.5 pour IQR.
        
    Retourne:
        tuple: Limites inférieure et supérieure pour les valeurs aberrantes.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound


def remove_outliers(X_train, y_train, num_features, factor=1.5, log_output=True):
    """
    Supprime les valeurs aberrantes des caractéristiques numériques spécifiées.
    
    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        y_train (Series ou DataFrame): Cible d'entraînement.
        num_features (list): Liste des noms de colonnes numériques à vérifier pour les outliers.
        factor (float): Facteur d'IQR pour délimiter les outliers.
        log_output (bool): Active les logs détaillés si True (par défaut).
        
    Retourne:
        tuple: (X_train, y_train) filtrés sans outliers.
    """
    if log_output:
        logging.info("Début du filtrage des outliers.")
    
    mask = pd.Series(True, index=X_train.index)  # Initialise le masque global
    for col in num_features:
        lower_bound, upper_bound = delimiter_outliers(X_train, col, factor=factor)
        col_mask = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
        mask &= col_mask  # Combine les masques pour chaque colonne
        
        if log_output:
            logging.info(f"Outliers supprimés pour {col}. Limites: [{lower_bound}, {upper_bound}]")

    # Filtrage final
    return X_train[mask], y_train[mask]


def remove_var_missing_values(X_train, X_test):
    """
    Supprime les colonnes ayant un taux de valeurs manquantes supérieur au seuil défini.
    
    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        X_test (DataFrame): Données de test.
        missing_ratio_threshold (float): Seuil de valeurs manquantes, par défaut 30%.
        
    Retourne:
        tuple: (X_train, X_test) filtrés et liste des colonnes supprimées.
    """
    missing_ratio_threshold=0.3
    missing_ratios = X_train.isnull().mean()
    cols_to_keep = missing_ratios[missing_ratios < missing_ratio_threshold].index
    removed_columns = missing_ratios[missing_ratios >= missing_ratio_threshold].index
    
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]
    
    logging.info(f"{len(removed_columns)} colonnes supprimées (> {missing_ratio_threshold*100}% manquants) :\n{removed_columns.tolist()}")
    return X_train, X_test


def reg_codes_postaux(X_train, col='code_postal_ban', threshold_code_postal=200):
    """
    Regroupe les codes postaux peu fréquents dans une catégorie 'Autres'.
    
    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        col (str): Nom de la colonne des codes postaux.
        threshold_code_postal (int): Seuil de fréquence pour conserver un code postal.
        
    Retourne:
        DataFrame: Données avec codes postaux rares regroupés.
    """
    if col in X_train.columns:
        frequent_codes = X_train[col].value_counts()[lambda x: x > threshold_code_postal].index
        X_train[col] = X_train[col].where(X_train[col].isin(frequent_codes), 'Autres')
    return X_train


def reg_types_energie(X_train, threshold_energy=100):
    """
    Regroupe les types d'énergie peu fréquents sous une étiquette 'Autres'.
    
    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        threshold_energy (int): Seuil de fréquence pour conserver un type d'énergie.
        
    Retourne:
        DataFrame: Données avec types d'énergie rares regroupés.
    """
    if 'type_energie_n_1' in X_train.columns:
        frequent_energies = X_train['type_energie_n_1'].value_counts()[lambda x: x >= threshold_energy].index
        X_train['type_energie_n_1'] = X_train['type_energie_n_1'].where(X_train['type_energie_n_1'].isin(frequent_energies), 'Autres')
    return X_train


def destroy_indexes(X_train, X_test, y_train, y_test):
    """
    Supprime la colonne 'geopoint' si elle est présente et réinitialise les index.
    
    Paramètres:
        X_train, X_test (DataFrame): Données d'entraînement et de test.
        y_train, y_test (Series ou DataFrame): Cibles d'entraînement et de test.
        
    Retourne:
        tuple: Données et cibles réinitialisées sans la colonne 'geopoint'.
    """
    for df in [X_train, X_test]:
        if 'geopoint' in df.columns:
            df.drop(columns=['geopoint'], inplace=True)
    
    # Réinitialisation des index
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


# --- Fonctions d'Optimisation et de Prétraitement --- 

def optimal_n_neighbors_for_knnimputer(X_train_quant, neighbors_to_test=[3, 5, 7, 10]):
    """
    Trouve le nombre optimal de voisins pour l'imputation KNN en utilisant la validation croisée
    et en minimisant le RMSE sur les valeurs manquantes.

    Paramètres:
        X_train_quant (DataFrame): Données quantitatives d'entraînement.
        neighbors_to_test (list): Liste de nombres de voisins à tester.

    Retourne:
        int: Meilleur nombre de voisins pour l'imputation KNN.
    """
    logger.info("Calcul du meilleur nombre de voisins pour KNNImputer.")
    rmse_scores = []
    scaler = StandardScaler()

    # Standardiser les données
    X_train_quant_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_quant),
        columns=X_train_quant.columns,
        index=X_train_quant.index
    )

    for n in neighbors_to_test:
        knn_imputer = KNNImputer(n_neighbors=n)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_fold = []

        for train_index, val_index in kf.split(X_train_quant_scaled):
            # Séparation en plis de validation croisée
            X_train_fold = X_train_quant_scaled.iloc[train_index]
            X_val_fold = X_train_quant_scaled.iloc[val_index]

            # Imputation et inverse de la standardisation
            X_train_imputed_scaled = pd.DataFrame(
                knn_imputer.fit_transform(X_train_fold),
                index=X_train_fold.index,
                columns=X_train_fold.columns
            )
            X_train_imputed = pd.DataFrame(
                scaler.inverse_transform(X_train_imputed_scaled),
                index=X_train_imputed_scaled.index,
                columns=X_train_imputed_scaled.columns
            )
            X_train_fold_orig = pd.DataFrame(
                scaler.inverse_transform(X_train_fold),
                index=X_train_fold.index,
                columns=X_train_fold.columns
            )

            # Calcul du RMSE pour les valeurs manquantes
            mask_missing = X_train_fold_orig.isna()
            rmse_per_column = []
            for col in X_train_fold.columns:
                original_values = X_train_fold_orig[col][mask_missing[col]]
                imputed_values = X_train_imputed[col][mask_missing[col]]
                if len(original_values) > 0:
                    rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))
                    rmse_per_column.append(rmse)
            rmse_fold.append(np.mean(rmse_per_column))

        # Ajouter le score RMSE moyen pour ce nombre de voisins
        rmse_scores.append((n, np.mean(rmse_fold)))

    # Meilleur nombre de voisins
    best_n = min(rmse_scores, key=lambda x: x[1])[0]
    logger.info(f"Le meilleur nombre de voisins pour KNNImputer est {best_n}.")
    return best_n


def pca_app(X_train, variance_threshold=0.9):
    """
    Applique l'analyse en composantes principales (ACP) pour déterminer le nombre de composants nécessaires 
    pour atteindre un seuil de variance cumulative, et trace les résultats avec le test des bâtons brisés.

    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        variance_threshold (float): Seuil de variance cumulative souhaité (par défaut 0.9 pour 90%).

    Retourne:
        int: Nombre optimal de composants principaux.
    """
    pca = PCA(svd_solver='full')
    pca.fit(X_train)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Calcul du test des bâtons brisés
    broken_stick_values = broken_stick_test(len(explained_variance))

    # Tracé de la variance expliquée vs test des bâtons brisés
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Variance Expliquée')
    plt.plot(range(1, len(explained_variance) + 1), broken_stick_values, label='Test des Bâtons Brisés', linestyle='--')
    plt.axvline(optimal_components, color='r', linestyle='--', label=f'{optimal_components} Composantes (pour {variance_threshold * 100:.0f}%)')
    plt.title('Courbe de la Variance Expliquée vs Test des Bâtons Brisés')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Expliquée')
    plt.legend()
    plt.grid()
    plt.show()

    return optimal_components


def broken_stick_test(num_components):
    """
    Calcule les valeurs de référence pour le test des bâtons brisés.

    Paramètres:
        num_components (int): Nombre de composants à tester.

    Retourne:
        list: Valeurs de référence pour chaque composant.
    """
    return [sum(1 / j for j in range(i, num_components + 1)) / num_components for i in range(1, num_components + 1)]


def preprocess_all_data(dfa, dfn):
    """
    Prépare et combine les données d'anciens et nouveaux logements, en appliquant diverses transformations
    et en créant de nouvelles caractéristiques.

    Paramètres:
        dfa (DataFrame): Données de logements anciens.
        dfn (DataFrame): Données de logements neufs.

    Retourne:
        tuple: DataFrames transformés (df, df_all) pour modélisation et analyse.
    """
    logger.info("Début du prétraitement des données.")
    
    # Ajout de l'indicateur 'logement' et imputation des années de construction
    dfa['logement'] = "ancien"
    dfn['logement'] = "neuf"
    current_year = pd.Timestamp.now().year
    dfn['Année_construction'] = current_year
    # Concaténation des DataFrames sur les colonnes communes
    col_communes = list(set(dfa.columns) & set(dfn.columns))
    df = pd.concat([dfa[col_communes], dfn[col_communes]], ignore_index=True)
    logger.info(f"DataFrame combiné avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    
    # Filtrage des enregistrements non nulls dans 'Conso_5_usages_é_finale'
    df.dropna(subset=['Conso_5_usages_é_finale'], inplace=True)

    # Création de la colonne 'passoire_energetique' selon l'étiquette DPE
    df["passoire_energetique"] = df['Etiquette_DPE'].isin(['F', 'G']).astype(str).map({'True': 'oui', 'False': 'non'})

    # Catégorisation des périodes de construction et nettoyage des données temporelles
    df['periode_construction'] = df['Année_construction'].apply(categoriser_annee_construction)
    df.drop(columns=['Année_construction'], inplace=True)
    df['Date_réception_DPE'] = pd.to_datetime(df['Date_réception_DPE'], errors='coerce')
    df['Timestamp_réception_DPE'] = df['Date_réception_DPE'].astype('int64') // 10**9
    df['Hauteur_sous-plafond'] = df['Hauteur_sous-plafond'].astype(float)
    
    df.drop(columns=['Date_réception_DPE'], inplace=True)

    # Conversion de certaines colonnes en chaînes pour cohérence
    df['Code_postal_(BAN)'] = df['Code_postal_(BAN)'].astype(str)
    df['N°_étage_appartement'] = df['N°_étage_appartement'].astype(str)

    # Nettoyage des noms de colonnes
    df = df.rename(columns=lambda x: clean_text(x))

    # Suppression de colonnes non actives et création du DataFrame final
    columns_to_exclude = ['geopoint']
    active_features = df.columns.difference(columns_to_exclude).tolist()
    df_all = df.copy()
    df = df[active_features]
    
    logger.info(f"DataFrame df réduit aux caractéristiques actives: {df.shape}.")
    return df, df_all

# --- Fonctions de Prétraitement et de Sélection de Caractéristiques ---

def preprocess_splitted_data(df, column_name, is_classification=True):
    """
    Divise les données en ensembles d'entraînement et de test, effectue divers prétraitements, et
    identifie les colonnes numériques et catégorielles.

    Paramètres:
        df (DataFrame): Le DataFrame à diviser et prétraiter.
        column_name (str): Le nom de la colonne cible.
        is_classification (bool): Indique si la tâche est de classification (True) ou de régression (False).

    Retourne:
        tuple: (X_train, X_test, y_train, y_test, num_features, cat_features)
    """
    # Séparation des caractéristiques (X) et de la cible (y)
    X = df.drop(columns=[column_name])
    y = df[column_name]

    # Option de stratification si classification
    stratify_option = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    # Gestion de la colonne 'n_dpe' pour correspondre aux index si présente
    if 'n_dpe' in X_train.columns:
        X_train.set_index('n_dpe', inplace=True)
        X_test.set_index('n_dpe', inplace=True)
        y_train.index = X_train.index
        y_test.index = X_test.index

    # Séparation des caractéristiques numériques et catégorielles
    num_features = X_train.select_dtypes(include=['float64', 'int']).columns
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Nettoyage des valeurs manquantes et recalcul des caractéristiques
    X_train, X_test = remove_var_missing_values(X_train, X_test)
    num_features = X_train.select_dtypes(include=['float64', 'int']).columns
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Suppression des valeurs aberrantes et transformations spécifiques
    X_train, y_train = remove_outliers(X_train, y_train, num_features)
    X_train = reg_codes_postaux(X_train)
    X_train = reg_types_energie(X_train)

    return X_train, X_test, y_train, y_test, num_features, cat_features


def preprocess_pipeline(X_train, num_features, cat_features):
    """
    Construit un pipeline de prétraitement avec imputation et normalisation pour les variables numériques
    et encodage pour les variables catégorielles.

    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        num_features (Index): Colonnes numériques de X_train.
        cat_features (Index): Colonnes catégorielles de X_train.

    Retourne:
        ColumnTransformer: Pipeline de prétraitement complet pour l'application aux données.
    """
    # Imputation KNN avec le meilleur nombre de voisins
    X_train_quant = X_train[num_features]
    best_n_neighbors = optimal_n_neighbors_for_knnimputer(X_train_quant)

    # Pipeline pour les caractéristiques numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=best_n_neighbors)),
        ('scaler', StandardScaler())
    ])

    # Pipeline pour les caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    # Assemblage du transformateur de colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    return preprocessor


def feature_selection_classification(X_train, y_train):
    """
    Sélectionne les caractéristiques importantes pour une tâche de classification en utilisant un modèle
    RandomForestClassifier.

    Paramètres:
        X_train (DataFrame): Données d'entraînement.
        y_train (Series): Étiquettes d'entraînement.

    Retourne:
        tuple: (X_train_reduced, selector) où X_train_reduced est X_train réduit aux caractéristiques sélectionnées.
    """
    # Initialisation et ajustement du RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)

    # Transformation pour conserver uniquement les caractéristiques importantes
    X_train_reduced = selector.transform(X_train)

    return X_train_reduced, selector

model = joblib.load('src/utils/models/classifier_model_v0.joblib')

