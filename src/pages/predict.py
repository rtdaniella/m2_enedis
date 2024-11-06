import html
from src.utils.models.classifier_model import load_classifier
from src.utils.models.regressor_model import load_regressor


classifier = load_classifier()
regressor = load_regressor()

# Layout de l'application Dash
predict.layout = html.Div([
    html.H1("Prédiction DPE et Consommation d'Énergie"),
    
    # Interface utilisateur pour la classification
    dcc.Dropdown(id='classification-dropdown', options=[
        {'label': 'Option 1', 'value': 'value1'},
        # Ajouter d'autres options selon ton besoin
    ]),
    
    # Autres éléments de l'interface utilisateur

])

# Callbacks pour interagir avec les modèles de machine learning
@predict.callback(
    Output('classification-output', 'children'),
    [Input('classification-dropdown', 'value')]
)
def update_classification(value):
    # Charger les données d'entrée et appliquer le prétraitement
    df = pd.read_csv('data.csv')
    X_class, y_class, X_reg, y_reg = preprocess_data(df)
    
    # Utiliser le modèle de classification pour faire une prédiction
    y_pred_class = classifier.predict(X_class)
    
    return f"Résultat classification: {y_pred_class}"