from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
# Importer les composants spécifiques à chaque page de l'application
from components.navbar import create_sidenav
from components.footer import create_footer
from pages.home import create_home_page
from pages.about import create_about_page
from pages.context import create_context_page 
from pages.charts import create_charts_page  
from pages.map import create_map_page  
from pages.not_found_404 import create_not_found_page
from pages.prediction_dpe import create_pred_dpe_page
from pages.prediction_conso import create_pred_conso_page
from utils.models.predict_classifier import predict_classifier  # Fonction pour effectuer la prédiction
from utils.models.predict_regressor import predict_regressor  # Fonction pour effectuer la prédiction

# Création de l'application Dash
app = Dash(
    __name__,
    external_stylesheets=[  # Ajout des feuilles de style externes (Bootstrap et icônes)
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
    ],
    suppress_callback_exceptions=True,  # Permet d'éviter des erreurs si un callback n'est pas encore défini
)

# Définir le titre de l'application
app.title = "GreenTech Solutions"

# Définir la barre de navigation et le pied de page de l'application
navbar = create_sidenav()  # Composant de la barre de navigation latérale
footer = create_footer()  # Composant du pied de page

# Layout de l'application : structure HTML de la page
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),  # Permet de détecter l'URL courante et de charger la page appropriée
        navbar,  # Barre de navigation
        html.Div(id="page-content"),  # Contenu de la page affichée
        footer,  # Pied de page
    ]
)

# Callback pour afficher la page en fonction de l'URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    """
    Cette fonction gère l'affichage des différentes pages en fonction du chemin URL.
    Elle détermine quelle page afficher en fonction de l'URL demandée.
    """
    if pathname == "/":
        return create_home_page()  # Si la page d'accueil est demandée
    elif pathname == "/about":
        return create_about_page()  # Si la page "À propos" est demandée
    elif pathname == "/context":
        return create_context_page()  # Si la page de contexte est demandée
    elif pathname == "/charts":
         return create_charts_page()  # Si la page des graphiques est demandée
    elif pathname == "/map":
         return create_map_page()  # Si la page de la carte est demandée
    elif pathname == "/pred_dpe":
        return create_pred_dpe_page()  # Si la page de prédiction DPE est demandée
    elif pathname == "/pred_conso":
        return create_pred_conso_page()  # Si la page de prédiction consommation énergétique est demandée
    else:
        return create_not_found_page()  # Si l'URL ne correspond à aucune page, afficher la page 404

# Callback pour rediriger les boutons vers les pages de prédiction
@app.callback(
    Output("url", "pathname"),  # On modifie l'URL pour rediriger l'utilisateur
    [
        Input("button-1", "n_clicks"),
        Input("button-2", "n_clicks"),
    ],  # Suivi des clics sur les boutons
)
def update_url(button1_clicks, button2_clicks):
    """
    Cette fonction redirige l'utilisateur vers la page de prédiction DPE ou consommation énergétique
    en fonction du bouton cliqué.
    """
    if button1_clicks:  # Si le bouton "Prédiction étiquette DPE" est cliqué
        return "/pred_dpe"
    elif button2_clicks:  # Si le bouton "Prédiction consommation énergétique" est cliqué
        return "/pred_conso"
    return "/"  # Si aucun bouton n'est cliqué, retourner à la page d'accueil

# Callback pour la page de prédiction DPE
from pages.prediction_dpe import form_fields  # Importer les form_fields de la page de prédiction DPE pour les utiliser dans le callback
from pages.prediction_conso import form_fields_conso 
@app.callback(
    Output("prediction-result-dpe", "children"),  # On met à jour le texte de la prédiction
    Input("predict-button-dpe", "n_clicks"),
    [Input(f"{column}-input", "value") for column in form_fields]  # Récupérer toutes les valeurs des champs du formulaire
)
def update_prediction_dpe(n_clicks, *inputs):
    """
    Cette fonction est appelée lorsque l'utilisateur clique sur le bouton "Prédire".
    Elle récupère les valeurs saisies dans le formulaire, appelle la fonction de prédiction et retourne le résultat.
    """
    if n_clicks is None:  # Si aucun clic sur le bouton n'a eu lieu
        return ""
    
    # Appeler la fonction de prédiction en lui passant les entrées utilisateur
    result = predict_classifier(inputs)
    
    return result  # Retourner le résultat de la prédiction pour l'afficher dans l'interface


@app.callback(
    Output("prediction-result-conso", "children"),  # On met à jour le texte de la prédiction
    Input("predict-button-conso", "n_clicks"),
    [Input(f"{column}-input", "value") for column in form_fields_conso]  # Récupérer toutes les valeurs des champs du formulaire
)
def update_prediction_conso(n_clicks, *inputs):
    """
    Cette fonction est appelée lorsque l'utilisateur clique sur le bouton "Prédire".
    Elle récupère les valeurs saisies dans le formulaire, appelle la fonction de prédiction et retourne le résultat.
    """
    if n_clicks is None:  # Si aucun clic sur le bouton n'a eu lieu
        return ""
    
    # Appeler la fonction de prédiction en lui passant les entrées utilisateur
    result = predict_regressor(inputs)
    
    return result  # Retourner le résultat de la prédiction pour l'afficher dans l'interface

# Exécution de l'application Dash
if __name__ == "__main__":
    app.run_server(debug=False)  # Lancer le serveur Dash en mode debug pour faciliter le développement
