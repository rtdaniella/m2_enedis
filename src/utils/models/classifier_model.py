import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.utils.models.preprocess import destroy_indexes, pca_app, preprocess_all_data, preprocess_pipeline, preprocess_splitted_data
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score

def train_classifier(X_train, y_train, X_test, y_test):
    # Import des données DPE sous forme de dataframes
    dfa = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-existants.csv')
    dfn = pd.read_csv('https://raw.githubusercontent.com/rtdaniella/m2_enedis/refs/heads/develop/src/files/dpe-v2-logements-neufs.csv')

    #dfa et dfn sont les dataframes de l'ancien et du neuf
    df,df_all = preprocess_all_data(dfa,dfn)
    X_train, X_test, y_train, y_test,num_features,cat_features =  preprocess_splitted_data(df,'etiquette_dpe')
    destroy_indexes(X_train,X_test,y_train,y_test)
    optimal_components = pca_app(X_train)
    preprocessor = preprocess_pipeline(X_train,num_features,cat_features)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    ('smote', BorderlineSMOTE(random_state=42)),  # SMOTE pour équilibrer les classes
    ('pca', PCA(svd_solver='full')), 
    ('classifier', RandomForestClassifier())
    ])

    param_grid = {
    'pca__n_components': [optimal_components,3,4],  # Nombre de composantes à tester
    'classifier__n_estimators': [50, 100, 200],  # Nombre d'estimateurs pour RandomForest
    'classifier__max_depth': [None, 10, 20, 30]  # Profondeur maximale de l'arbre
}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,scoring='f1_macro',
                           cv=StratifiedKFold(n_splits=5),
                           n_jobs=-1,
                           verbose=1)
    grid_search.fit(X_train, y_train)

    # Évaluation du modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Rapport de classification
    
    print(classification_report(y_test, y_pred))
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print('recall macro:' + str(recall_score(y_test, y_pred, average='macro')))
    print('precision macro: ' + str(precision_score(y_test, y_pred, average='macro')))
    print('recall micro: ' + str(recall_score(y_test, y_pred, average='micro')))
    print('precision micro: ' + str(precision_score(y_test, y_pred, average='micro')))
    # Sauvegarder le modèle
    joblib.dump(best_model, 'models/classifier_model.joblib')

def load_classifier():
    # Charger le modèle
    return joblib.load('models/classifier_model.joblib')