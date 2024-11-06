import pandas as pd
from dash import html, dash_table, dcc, Input, Output
import dash
import dash_daq as daq

# Charger les données CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Créer la page de contexte avec tableau et filtres
def create_context_page():
    # Charger les données depuis un fichier CSV
    data = load_data("C:/Users/danie/OneDrive/Documents/GitHub/m2_enedis/src/files/dpe-nettoye.csv")

    # Récupérer les colonnes du CSV pour le tableau
    columns = [{'name': col, 'id': col} for col in data.columns]
    
    # Récupérer les valeurs uniques de chaque colonne pour les filtres
    dpe_values = data['etiquette_dpe'].unique()
    periode_construction_values = data['periode_construction'].unique()
    type_energie_values = data['type_energie_n_1'].unique()
    code_postal_values = data['code_postal_ban'].unique()

    # Exemple de KPI pour les cartes
    total_entries = len(data)
    average_dpe = data['etiquette_dpe'].value_counts().idxmax()  # Exemple de calcul
    total_postals = len(data['code_postal_ban'].unique())
    
    return html.Div([

        # Conteneur des cartes KPI
        html.Div([
            # Première carte KPI
            html.Div([
                html.Div([
                    html.I(className="fas fa-house-user", style={'fontSize': '40px', 'color': 'white'}),
                    html.H5("Total des Entrées", style={'color': 'white', 'marginTop': '10px'}),
                    html.P(f"{total_entries}", style={'color': 'white', 'fontSize': '24px', 'fontWeight': 'bold'})
                ], style={
                    'backgroundColor': '#4CAF50',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'height': '170px',
                    'width': '250px',  # Assurez-vous que toutes les cartes ont la même largeur
                    'textAlign': 'center',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'margin': '10px',
                    'display': 'inline-block',
                }),
            ], style={'display': 'inline-block'}),  # Alignement horizontal

            # Deuxième carte KPI
            html.Div([
                html.Div([
                    html.I(className="fas fa-calendar-alt", style={'fontSize': '40px', 'color': 'white'}),
                    html.H5("DPE Maximum", style={'color': 'white', 'marginTop': '10px'}),
                    html.P(f"{average_dpe}", style={'color': 'white', 'fontSize': '24px', 'fontWeight': 'bold'})
                ], style={
                    'backgroundColor': '#FF9800',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'height': '170px',
                    'width': '250px',  # Assurez-vous que toutes les cartes ont la même largeur
                    'textAlign': 'center',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'margin': '10px',
                    'display': 'inline-block',
                }),
            ], style={'display': 'inline-block'}),  # Alignement horizontal

            # Troisième carte KPI
            html.Div([
                html.Div([
                    html.I(className="fas fa-map-marker-alt", style={'fontSize': '40px', 'color': 'white'}),
                    html.H5("Total des Codes Postaux", style={'color': 'white', 'marginTop': '10px'}),
                    html.P(f"{total_postals}", style={'color': 'white', 'fontSize': '24px', 'fontWeight': 'bold'})
                ], style={
                    'backgroundColor': '#2196F3',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'height': '170px',
                    'width': '250px',  # Assurez-vous que toutes les cartes ont la même largeur
                    'textAlign': 'center',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'margin': '10px',
                    'display': 'inline-block',
                }),
            ], style={'display': 'inline-block'}),  # Alignement horizontal
        ], style={'textAlign': 'center', 'marginBottom': '30px', 'display': 'flex', 'justifyContent': 'center'}),  # Centrer les cartes et espace en bas

        # Conteneur des filtres avec alignement horizontal de deux colonnes
        html.Div([
            # Première colonne (deux filtres verticaux)
            html.Div([ 
                html.Div([
                    html.Label('Filtrer par étiquette DPE', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='dpe-filter',
                        options=[{'label': val, 'value': val} for val in dpe_values],
                        multi=True,  # Permet de sélectionner plusieurs valeurs
                        placeholder="Sélectionner une étiquette DPE",
                        style={'width': '100%', 'padding': '5px', 'height': '30px'}  # Réduction de la hauteur
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label('Filtrer par période de construction', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='periode-construction-filter',
                        options=[{'label': val, 'value': val} for val in periode_construction_values],
                        multi=True,
                        placeholder="Sélectionner une période de construction",
                        style={'width': '100%', 'padding': '5px', 'height': '30px'}
                    ),
                ], style={'marginBottom': '20px'})
            ], style={'width': '48%', 'marginRight': '4%'}),  # Première colonne avec largeur de 48%

            # Deuxième colonne (deux filtres verticaux)
            html.Div([
                html.Div([
                    html.Label('Filtrer par type d\'énergie', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='type-energie-filter',
                        options=[{'label': val, 'value': val} for val in type_energie_values],
                        multi=True,
                        placeholder="Sélectionner un type d'énergie",
                        style={'width': '100%', 'padding': '5px', 'height': '30px'}
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label('Filtrer par code postal', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='code-postal-filter',
                        options=[{'label': val, 'value': val} for val in code_postal_values],
                        multi=True,
                        placeholder="Sélectionner un code postal",
                        style={'width': '100%', 'padding': '5px', 'height': '30px'}
                    ),
                ])
            ], style={'width': '48%'}),  # Deuxième colonne avec largeur de 48%
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),  # Aligner les deux colonnes horizontalement

        # Ajouter un espace entre les filtres et le tableau
        html.Div(style={'height': '20px'}),  # Espace de 20px

        # Tableau de données
        dash_table.DataTable(
            id="data-table",
            columns=columns,
            data=data.to_dict('records'),
            style_as_list_view=True,
            style_table={
                'overflowX': 'auto',
                'maxWidth': '100%',
                'maxHeight': '570px',  # Limite la hauteur et active le défilement vertical
                'border': '1px solid black',  # Bordure du tableau
            },
            style_cell={
                'backgroundColor': 'lightgrey',  # Couleur de fond des cellules
                'color': 'black',  # Couleur du texte
                'textAlign': 'left',
                'padding': '10px',
                'minWidth': '120px', 'width': '150px', 'maxWidth': '200px',  # Ajuste la largeur des colonnes
                'whiteSpace': 'normal',  # Permet le retour à la ligne si le texte est trop long
                'border': '1px solid black',  # Bordure des cellules
            },
            style_header={
                'backgroundColor': 'grey',  # Fond de l'en-tête
                'fontWeight': 'bold',
                'color': 'white',
                'border': '1px solid black',  # Bordure de l'en-tête
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'lightgrey'},  # Couleur de fond pour les lignes impaires
            ],
            style_cell_conditional=[
                {'if': {'column_id': col}, 'width': '200px'} for col in data.columns
            ]
        )
    ],
        style={
            "padding": "20px",
            "marginLeft": "260px"  # Décalage pour laisser de l'espace à la SideNav
        })

# Configuration de l'application Dash
app = dash.Dash(__name__)

# Mettre à jour le tableau en fonction des filtres
@app.callback(
    Output('data-table', 'data'),
    [
        Input('dpe-filter', 'value'),
        Input('periode-construction-filter', 'value'),
        Input('type-energie-filter', 'value'),
        Input('code-postal-filter', 'value')
    ]
)
def update_table(dpe_filter, periode_filter, energie_filter, postal_filter):
    # Charger les données depuis un fichier CSV
    data = load_data("C:/Users/danie/OneDrive/Documents/GitHub/m2_enedis/src/files/dpe-nettoye.csv")
    
    # Appliquer les filtres si sélectionnés
    if dpe_filter:
        data = data[data['etiquette_dpe'].isin(dpe_filter)]
    if periode_filter:
        data = data[data['periode_construction'].isin(periode_filter)]
    if energie_filter:
        data = data[data['type_energie_n_1'].isin(energie_filter)]
    if postal_filter:
        data = data[data['code_postal_ban'].isin(postal_filter)]

    return data.to_dict('records')

# Définir la mise en page de l'application
app.layout = create_context_page()

if __name__ == "__main__":
    app.run_server(debug=True)
