import logging
import os
from venv import logger
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


def categoriser_annee_construction(valeur):
    
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

def replace_chars(text):
    # Remplacer les accents
    for accent, sans_accent in accents.items():
        text = text.replace(accent, sans_accent)

    # Remplacer les caractères spéciaux par un underscore
    for char in special_chars:
        text = text.replace(char, '_')

    # Supprimer les underscores consécutifs
    while '__' in text:
        text = text.replace('__', '_')

    # Supprimer les underscores au début et à la fin d'une chaine
    return text.strip('_').lower()

def clean_string(text):
    # Remplacer les accents
    for accent, sans_accent in accents.items():
        text = text.replace(accent, sans_accent)

    # Remplacer les caractères spéciaux par un underscore
    for char in special_chars:
        text = text.replace(char, '_')

    # Supprimer les underscores consécutifs
    while '__' in text:
        text = text.replace('__', '_')

    # Supprimer les underscores au début et à la fin et mettre en minuscules
    return text.strip('_').lower()

def delimiter_outliers(df,col, factor=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound

def remove_outliers(X_train, y_train, num_features, factor=1.5):
    logger.info("Début du filtrage des outliers.")
    
    # Masque général pour conserver uniquement les lignes sans outliers
    mask = pd.Series(True, index=X_train.index)

    for col in num_features:
        lower_bound, upper_bound = delimiter_outliers(X_train, col, factor=factor)
        col_mask = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
        mask &= col_mask  # Combine les masques pour chaque colonne
        
        logger.info(f"Outliers supprimés pour la colonne {col}. Limites: [{lower_bound}, {upper_bound}]")

    # Filtrer les lignes sans outliers
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    logger.info(f"Filtrage effectué. Nouveau nombre de lignes dans X_train : {X_train.shape[0]}")
    return X_train, y_train


def remove_var_missing_values(X_train, X_test):
# Seuil de valeurs manquantes à 30 %
    missing_ratio_threshold = 0.3
    missing_ratios = X_train.isnull().mean()

    # Colonnes à conserver selon le seuil
    cols_to_keep = missing_ratios[missing_ratios < missing_ratio_threshold].index
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]

    removed_columns = missing_ratios[missing_ratios >= missing_ratio_threshold].index
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]
    logging.info("{} colonnes ont été supprimées car le pourcentage de valeurs manquantes excédait {}% :\n{}".format(
        len(removed_columns), missing_ratio_threshold * 100, removed_columns.tolist()))    
    return X_train, X_test

def reg_codes_postaux(X_train,threshold_code_postal = 200):
    # Regroupement des codes postaux rares
    if 'code_postal_ban' in X_train.columns:
        # Calcul des codes fréquents selon le seuil
        frequent_codes = X_train['code_postal_ban'].value_counts()[lambda x: x > threshold_code_postal].index
        X_train['code_postal_ban'] = X_train['code_postal_ban'].where(X_train['code_postal_ban'].isin(frequent_codes), 'Autres')
    return X_train
    
def reg_types_energie(X_train, threshold_energy=100):    
    # Regroupement des types d'énergie
    frequent_energies = X_train['type_energie_n_1'].value_counts()[lambda x: x >= threshold_energy].index
    X_train['type_energie_n_1'] = X_train['type_energie_n_1'].where(X_train['type_energie_n_1'].isin(frequent_energies), 'Autres')
    return X_train

def destroy_indexes(X_train,X_test,y_train,y_test):
    if 'geopoint' in X_train.columns:
        X_train = X_train.drop(columns=['geopoint'])
    if 'geopoint' in X_test.columns:
        X_test = X_test.drop(columns=['geopoint'])
    
    # Réinitialisation des index
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def optimal_n_neighbors_for_knnimputer(X_train_quant, neighbors_to_test=[3, 5, 7, 10]):
    logger.info("Calcul du meilleur nombre de voisins pour KNNImputer.")

    rmse_scores = []
    scaler = StandardScaler()

    # Standardiser les données quantitatives
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
            X_train_fold = X_train_quant_scaled.iloc[train_index]
            X_val_fold = X_train_quant_scaled.iloc[val_index]

            # Imputer les valeurs manquantes
            X_train_imputed_scaled = pd.DataFrame(
                knn_imputer.fit_transform(X_train_fold),
                index=X_train_fold.index,
                columns=X_train_fold.columns
            )

            # Inverse de la standardisation
            X_train_imputed = pd.DataFrame(
                scaler.inverse_transform(X_train_imputed_scaled),
                index=X_train_imputed_scaled.index,
                columns=X_train_imputed_scaled.columns
            )
            X_train_fold_orig = scaler.inverse_transform(X_train_fold)
            X_train_fold_orig = pd.DataFrame(X_train_fold_orig, index=X_train_fold.index, columns=X_train_fold.columns)

            # Calcul du RMSE uniquement sur les valeurs manquantes initiales
            mask_missing = X_train_fold_orig.isna()

            # Calculer le RMSE par colonne
            rmse_per_column = []
            for col in X_train_fold.columns:
                original_values = X_train_fold_orig[col][mask_missing[col]]
                imputed_values = X_train_imputed[col][mask_missing[col]]

                if len(original_values) > 0:
                    rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))
                    rmse_per_column.append(rmse)

            # Moyenne du RMSE pour toutes les colonnes de ce pli
            rmse_fold.append(np.mean(rmse_per_column))

        # Moyenne des RMSE des 5 plis
        rmse_scores.append((n, np.mean(rmse_fold)))

    # Sélection du meilleur nombre de voisins
    best_n = min(rmse_scores, key=lambda x: x[1])[0]
    logger.info(f"Le meilleur nombre de voisins pour KNNImputer est {best_n}.")
    return best_n

def pca_app(X_train, variance_threshold=0.9):
  pca = PCA(svd_solver='full')
  pca.fit(X_train)
# Variance expliquée par chaque composante
  explained_variance = pca.explained_variance_ratio_
  cumulative_variance = np.cumsum(explained_variance)
  optimal_components = np.argmax(cumulative_variance >= 0.9) + 1  # Nombre de composants pour 90%

# Calculer les valeurs du test des bâtons brisés
  broken_stick_values = broken_stick_test(len(explained_variance))

# Tracer la courbe de la variance expliquée et du test des bâtons brisés
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
    broken_stick_values = [sum(1 / j for j in range(i, num_components + 1)) / num_components for i in range(1, num_components + 1)]
    return broken_stick_values



def preprocess_all_data(dfa, dfn):
    logger.info("Début du prétraitement des données.")
    
    # Ajout de la colonne 'logement' pour distinguer ancien/neuf
    logger.info("Ajout de la colonne 'logement' aux deux DataFrames.")
    dfa['logement'] = "ancien"
    dfn['logement'] = "neuf"
    
    # Remplacer les années de construction manquantes dans dfn par l'année actuelle
    current_year = pd.Timestamp.now().year
    logger.info(f"Remplacement des années de construction manquantes par l'année {current_year}.")
    dfn['Année_construction'] = current_year
    
    # Trouver les colonnes communes aux deux DataFrames
    col_communes = list(set(dfa.columns) & set(dfn.columns))
    logger.info(f"Colonnes communes aux deux DataFrames: {col_communes}")

    # Concaténation des DataFrames sur les colonnes communes
    df = pd.concat([dfa[col_communes], dfn[col_communes]], ignore_index=True)
    logger.info(f"DataFrame combiné avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    df = df.dropna(subset=['Conso_5_usages_é_finale'])
    # Création de la colonne 'passoire_energetique'
    df["passoire_energetique"] = df['Etiquette_DPE'].isin(['F', 'G']).astype(str).map({'True': 'oui', 'False': 'non'})
    logger.info("Création de la colonne 'passoire_energetique' basée sur l'étiquette DPE.")

    # Catégoriser la période de construction
    df['periode_construction'] = df['Année_construction'].apply(categoriser_annee_construction)
    df.drop(columns=['Année_construction'], inplace=True)
    logger.info("Catégorisation des années de construction effectuée.")

    # Conversion des dates en datetime et en timestamp UNIX
    df['Date_réception_DPE'] = pd.to_datetime(df['Date_réception_DPE'], errors='coerce')
    df['Timestamp_réception_DPE'] = df['Date_réception_DPE'].astype('int64') // 10**9
    df.drop(columns=['Date_réception_DPE'], inplace=True)
    logger.info("Conversion des dates en timestamp UNIX.")

    # Conversion de certaines colonnes en chaînes de caractères pour uniformité
    df['Code_postal_(BAN)'] = df['Code_postal_(BAN)'].astype(str)
    df['N°_étage_appartement'] = df['N°_étage_appartement'].astype(str)
    logger.info("Conversion des colonnes spécifiques en chaînes de caractères effectuée.")
    
    # Nettoyage des noms de colonnes pour enlever accents et caractères spéciaux
    df = df.rename(columns=lambda x: replace_chars(x))
    

    # Exclusion de colonnes non actives (comme geopoint)
    columns_to_exclude = ['geopoint']
    logger.info(f"Colonnes après nettoyage : {df.columns}")
    active_features = df.columns.difference(columns_to_exclude).tolist()
    logger.info(df.isnull().sum())
    # Création de df_all pour conserver toutes les données et df pour le modèle
    df_all = df.copy()
    df = df[active_features]
    logger.info(f"DataFrame df réduit aux caractéristiques actives.,{df.shape}")

    return df, df_all


def preprocess_splitted_data(df,column_name, is_classification=True):

    X = df.drop(columns=[column_name])  
    y = df[column_name]                  

    stratify_option = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    if 'n_dpe' in X_train.columns:
        X_train.set_index('n_dpe', inplace=True)
        X_test.set_index('n_dpe', inplace=True)
        y_train.index = X_train.index 
        y_test.index = X_test.index 

    num_features = X_train.select_dtypes(include=['float64', 'int']).columns
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns

    X_train, X_test = remove_var_missing_values(X_train, X_test)
    # Recalculation des caractéristiques numériques et catégorielles après nettoyage
    num_features = X_train.select_dtypes(include=['float64', 'int']).columns
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns

    X_train, y_train = remove_outliers(X_train, y_train,num_features)
    X_train = reg_codes_postaux(X_train)
    X_train = reg_types_energie(X_train)

    return X_train, X_test, y_train, y_test,num_features,cat_features


def preprocess_pipeline(X_train,num_features,cat_features):
# Extraction des colonnes quantitatives
    X_train_quant = X_train[num_features]

# Calcul du meilleur nombre de voisins pour le KNNImputer
    best_n_neighbors = optimal_n_neighbors_for_knnimputer(X_train_quant)

    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=best_n_neighbors)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False, drop='first'))
        
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    return preprocessor
# Définir la fonction de sélection de features
  

def feature_selection_classification(X_train, y_train):
     # Initialisation du RandomForestClassifier pour sélectionner les caractéristiques
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Création de SelectFromModel sans prefit=True
    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)  # entraînement du modèle ici

    # Transformer les données pour ne garder que les caractéristiques sélectionnées
    X_train_reduced = selector.transform(X_train)

    return X_train_reduced, selector

def feature_selection_regression(X_train, y_train):
    # Initialisation du RandomForestRegressor pour sélectionner les caractéristiques
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Entraînement du modèle de régression
    model.fit(X_train, y_train)

    # Sélection des caractéristiques importantes avec un seuil défini (par exemple, la moyenne)
    selector = SelectFromModel(model)

    X_train_reduced = selector.transform(X_train)

    # Retourner les données réduites et le modèle
    return X_train_reduced, selector


# def handle_missing_values_in_X_y_test(X_test, y_test, num_features, cat_features, threshold=0.05):
   
#     # Gestion des valeurs manquantes dans X_test
#     missing_values_count_X = X_test.isnull().sum()
#     print("Nombre de valeurs manquantes dans X_test :")
#     print(missing_values_count_X)

#     # Colonnes avec des valeurs manquantes au-dessus du seuil
#     columns_to_impute_X = missing_values_count_X[missing_values_count_X > X_test.shape[0] * threshold].index

#     # Séparer les colonnes numériques et catégorielles dans X_test
#     numeric_columns_to_impute = [col for col in columns_to_impute_X if col in num_features]
#     categorical_columns_to_impute = [col for col in columns_to_impute_X if col in cat_features]

#     # Imputation des colonnes numériques de X_test (par la moyenne)
#     if numeric_columns_to_impute:
#         print("Imputation des colonnes numériques dans X_test :")
#         print(numeric_columns_to_impute)
#         imputer_numeric = SimpleImputer(strategy='mean')
#         X_test[numeric_columns_to_impute] = imputer_numeric.fit_transform(X_test[numeric_columns_to_impute])

#     # Imputation des colonnes catégorielles de X_test (par la valeur la plus fréquente)
#     if categorical_columns_to_impute:
#         print("Imputation des colonnes catégorielles dans X_test :")
#         print(categorical_columns_to_impute)
#         imputer_categorical = SimpleImputer(strategy='most_frequent')
#         X_test[categorical_columns_to_impute] = imputer_categorical.fit_transform(X_test[categorical_columns_to_impute])

#     # Si aucune colonne de X_test ne dépasse le seuil, suppression des lignes avec NaN dans X_test
#     if columns_to_impute_X.empty:
#         print("Aucune colonne de X_test avec un nombre élevé de valeurs manquantes, suppression des lignes concernées.")
#         X_test = X_test.dropna()

#     # Gestion des valeurs manquantes dans y_test
#     if y_test.isnull().sum() > 0:
#         print("Valeurs manquantes détectées dans y_test.")
        
#         # Suppression des lignes de X_test et y_test où y_test est NaN
#         mask = y_test.notnull()
#         X_test = X_test[mask]
#         y_test = y_test[mask]
#         print(f"{(~mask).sum()} lignes supprimées en raison de valeurs manquantes dans y_test.")
        
#     # Vérification des valeurs manquantes après traitement
#     print("\nValeurs manquantes après traitement dans X_test :")
#     print(X_test.isnull().sum())
#     print("\nValeurs manquantes après traitement dans y_test :")
#     print(y_test.isnull().sum())

#     return X_test, y_test







# def preprocess_pipeline(X_train, num_features, cat_features, save_csv_path=None):
#     # Extraction des colonnes quantitatives
#     X_train_quant = X_train[num_features]

#     # Calcul du meilleur nombre de voisins pour le KNNImputer
#     best_n_neighbors = optimal_n_neighbors_for_knnimputer(X_train_quant)

#     numeric_transformer = Pipeline(steps=[
#         ('imputer', KNNImputer(n_neighbors=best_n_neighbors)) 
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, num_features),
#             ('cat', categorical_transformer, cat_features)
#         ]
#     )
    
#     # Appliquer le préprocesseur (sans appliquer l'encodeur OneHotEncoder)
#     # Cela inclut l'imputation et la mise à l'échelle mais sans encoder les variables catégorielles
#     X_train_transformed = preprocessor.fit_transform(X_train)

#     # Convertir les résultats après transformation en DataFrame
#     # Pour les variables numériques
#      # 1. Pour les variables numériques
#     X_train_num = pd.DataFrame(X_train_transformed[:, :len(num_features)], columns=num_features)
    
#     # 2. Pour les variables catégorielles
#     X_train_cat = pd.DataFrame(X_train_transformed[:, len(num_features):], columns=cat_features)

#     # Fusionner les deux DataFrames (numériques et catégoriques)
#     X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
#     if save_csv_path:
#         # Vérifiez si le répertoire existe, sinon, créez-le
#         directory = os.path.dirname(save_csv_path)
#         if directory and not os.path.exists(directory):
#             os.makedirs(directory)
#             print(f"Répertoire créé : {directory}")
        
#         # Sauvegarder le fichier CSV
#         X_train_final.to_csv(save_csv_path, index=False)
#         print(f"Jeu de données sauvegardé dans {save_csv_path}")
#     # Retourner le préprocesseur pour pouvoir l'utiliser dans un pipeline complet
#     return preprocessor