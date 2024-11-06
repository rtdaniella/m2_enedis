import datetime
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    """
    Fonction pour catégoriser les années de construction.

    Paramètre :
    valeur : int : Année de construction.

    Retourne :
    str : Période de construction.
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

def delimiter_outliers(df,col):
    """
    Déterminer les outliers pour une colonne donnée.
    On se base sur la méthode de l'écart interquartile (IQR).

    Paramètre :
    col : str : nom de la colonne

    Retourne :
    lower_bound : float : limite inférieure
    upper_bound : float : limite supérieure
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def remove_outliers(X_train, y_train,num_features):
    for col in num_features:
        lower_bound, upper_bound = delimiter_outliers(X_train, col)
        mask = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
        X_train = X_train[mask]
        y_train = y_train[mask]
    return X_train, y_train

def remove_var_missing_values(X_train, X_test):
# On autorise au maximum 30 % de valeurs manquantes
    missing_ratio_threshold = 0.3
    missing_ratios = X_train.isnull().mean()
    cols_to_keep = missing_ratios[missing_ratios < missing_ratio_threshold].index
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]
    diff = missing_ratios[missing_ratios >= missing_ratio_threshold].index
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]
    print("{} colonnes ont été supprimées car le pourcentage de valeurs manquantes excédait {}% :\n{}".format(len(diff), missing_ratio_threshold * 100, diff.tolist()))

def reg_codes_postaux(X_train):
    # Regroupement des codes postaux rares
    threshold_code_postal = 200
    frequent_codes = X_train['code_postal_ban'].value_counts()[X_train['code_postal_ban'].value_counts() > threshold_code_postal].index
    X_train['code_postal_ban'] = X_train['code_postal_ban'].where(X_train['code_postal_ban'].isin(frequent_codes), 'Autres')
    
def reg_types_energie(X_train):    
    # Regroupement des types d'énergie
    threshold_energy = 100
    energy_counts = X_train['type_energie_n_1'].value_counts()
    X_train['type_energie_n_1'] = X_train['type_energie_n_1'].where(X_train['type_energie_n_1'].isin(energy_counts[energy_counts >= threshold_energy].index), 'Autres')

def destroy_indexes(X_train,X_test,y_train,y_test):
    X_train.drop(columns=['geopoint'], inplace=True)
    X_test.drop(columns=['geopoint'], inplace=True)
# Réinitialiser les index de X_train et X_test
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

# Réinitialiser les index de y_train et y_test
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

def optimal_n_neighbors_for_knnimputer(X_train_quant, neighbors_to_test=[3, 5, 7, 10]):
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

            # Imputer les valeurs manquantes sur les données d'entraînement
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
    return best_n

def pca_app(X_train):
  pca = PCA(svd_solver='full')
  pca.fit(X_train)
# Variance expliquée par chaque composante
  explained_variance = pca.explained_variance_ratio_
  cumulative_variance = np.cumsum(explained_variance)
  num_components = len(explained_variance)
  optimal_components = np.argmax(cumulative_variance >= 0.9) + 1  # Nombre de composants pour 90%

# Calculer les valeurs du test des bâtons brisés
  broken_stick_values = broken_stick_test(num_components)

# Tracer la courbe de la variance expliquée et du test des bâtons brisés
  plt.figure(figsize=(10, 6))
  plt.plot(range(1, num_components + 1), explained_variance, marker='o', label='Variance Expliquée')
  plt.plot(range(1, num_components + 1), broken_stick_values, label='Test des Bâtons Brisés', linestyle='--')
  plt.title('Courbe de la Variance Expliquée vs Test des Bâtons Brisés')
  plt.xlabel('Nombre de Composantes')
  plt.ylabel('Variance Expliquée')
  plt.xticks(range(1, num_components + 1))
  plt.legend()
  plt.grid()
  plt.show()

  return optimal_components

def broken_stick_test(num_components):
    """ Calcule les valeurs du test des bâtons brisés. """
    return [(i / num_components) for i in range(1, num_components + 1)]

# Nombre maximum de composantes


def preprocess_all_data(dfa,dfn):
    dfa['logement'] = "ancien"
    dfn['logement'] = "neuf" 
    dfn['Année_construction'] = datetime.now().year
    # Déterminer les colonnes communes de neuf et ancien
    col_communes = list(set(dfa.columns) & set(dfn.columns))
    # print("Colonnes communes:", col_communes)

    # Réaliser un concat pour empiler les deux df neuf et ancien
    df = pd.concat([dfa,dfn], keys=col_communes, join="inner",ignore_index=True)
    df["passoire_energetique"] = df['Etiquette_DPE'].isin(['F', 'G']).map({True: 'oui', False: 'non'})
    df['periode_construction'] = df['Année_construction'].apply(categoriser_annee_construction)
    df.drop(columns=['Année_construction'], inplace=True)

    # Conversion des colonnes de date en datetime
    df['Date_réception_DPE'] = pd.to_datetime(df['Date_réception_DPE'])

    # Conversion en timestamps (en secondes depuis l'époque UNIX)
    df['Timestamp_réception_DPE'] = df['Date_réception_DPE'].astype('int64') // 10**9


    # Suppression des colonnes de date originales
    df.drop(columns=['Date_réception_DPE'], inplace=True)

    df['Code_postal_(BAN)'] = df['Code_postal_(BAN)'].astype(str)
    df['N°_étage_appartement'] = df['N°_étage_appartement'].astype(str)
    # Renommer les colonnes de df
    df = df.rename(columns=lambda x: replace_chars(x))
    print(df.columns)
    columns_to_exclude = ['geopoint']

    # Créer une liste des variables actives
    active_features = df.columns.difference(columns_to_exclude).tolist()
    df_all = df.copy()
    df = df[active_features]
 
    return df, df_all

def preprocess_splitted_data(df,column_name):

    X = df.drop(columns=[column_name])  
    y = df[column_name]                  

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train.set_index('n_dpe', inplace=True)
    X_test.set_index('n_dpe', inplace=True)
    y_train.index = X_train.index 
    y_test.index = X_test.index 

    num_features = X_train.select_dtypes(include=['float64', 'int']).columns
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns

    remove_var_missing_values(X_train, y_train)
    X_train, y_train = remove_outliers(X_train, y_train,num_features)
    reg_codes_postaux(X_train)
    reg_types_energie(X_train)

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
            ('cat', categorical_transformer, cat_features),
            ('selector',RandomForestClassifier(n_estimators=100,random_state=42))
        ]
    )

    return preprocessor


    
