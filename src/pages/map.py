import os
import sys
from dash import html, dcc, Input, Output
import pandas as pd
import folium
from folium.plugins import MarkerCluster
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.models.preprocess import preprocess_all_data

# Fonction pour générer la carte
def generate_map(df_filtered):
    # Créer la carte initiale centrée sur la moyenne des coordonnées des logements filtrés
    if not df_filtered.empty:
        map_center = [df_filtered['latitude'].mean(), df_filtered['longitude'].mean()]
    else:
        map_center = [df_filtered['latitude'].mean(), df_filtered['longitude'].mean()]

    # Initialiser la carte Folium
    m = folium.Map(location=map_center, zoom_start=12)

    # Créer un cluster pour les marqueurs
    marker_cluster = MarkerCluster().add_to(m)

    # Ajouter les marqueurs pour les points filtrés (en limitant si nécessaire)
    for _, row in df_filtered.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['type_batiment']} - {row['logement']}"
        ).add_to(marker_cluster)

    # Convertir la carte Folium en HTML pour l'afficher dans Dash
    return m._repr_html_()

def create_map_page(dfa, dfn):
    # Charger les données d'entrée
    df, df_all = preprocess_all_data(dfa, dfn)

    # Filtrer les données une seule fois
    df = df.set_index('n_dpe')
    df_all = df_all.set_index('n_dpe')
    df_intersection = df.loc[df.index.intersection(df_all.index)]

    # Sélectionner les colonnes nécessaires
    df_intersection = df_intersection[['code_postal_ban', 'type_batiment', 'logement']]
    df_intersection['geopoint'] = df_all.loc[df_intersection.index, 'geopoint']

    # Séparer les colonnes nécessaires pour la carte
    df_intersection[['latitude', 'longitude']] = df_intersection['geopoint'].str.split(',', expand=True)
    df_intersection['latitude'] = df_intersection['latitude'].astype(float)
    df_intersection['longitude'] = df_intersection['longitude'].astype(float)

    # Créer une liste des codes postaux uniques pour le filtre
    unique_codes_postaux = df_intersection['code_postal_ban'].unique()
    code_postal_options = [{'label': str(code), 'value': str(code)} for code in unique_codes_postaux]

    default_code_postal = "69310" if "69310" in unique_codes_postaux else unique_codes_postaux[0]

    df_filtered = df_intersection[df_intersection['code_postal_ban'] == default_code_postal]

    # Créer la carte
    map_html = generate_map(df_filtered)

    return html.Div([
        html.H1("Carte des Logements DPE"),
        html.P("Sélectionnez un code postal pour filtrer les logements."),
        
        # Menu déroulant pour sélectionner un code postal
        dcc.Dropdown(
            id='postal-code-dropdown',
            options=code_postal_options,
            value=default_code_postal,  
            multi=False,  
            style={'width': '50%'}
        ),
        
        # Conteneur pour afficher la carte filtrée
        html.Div(id='map-container', children=[html.Iframe(srcDoc=map_html, width="100%", height="500px")])
    ])

# Callback pour mettre à jour la carte en fonction du code postal sélectionné
def register_callbacks(app, dfa, dfn):
    @app.callback(
        Output('map-container', 'children'),
        Input('postal-code-dropdown', 'value')
    )
    def update_map(selected_code_postal):
        # Charger et filtrer les données
        df, df_all = preprocess_all_data(dfa, dfn)
        df = df.set_index('n_dpe')
        df_all = df_all.set_index('n_dpe')
        df_intersection = df.loc[df.index.intersection(df_all.index)]
        df_intersection = df_intersection[['code_postal_ban', 'type_batiment', 'logement']]
        df_intersection['geopoint'] = df_all.loc[df_intersection.index, 'geopoint']
        
        # Séparer les géopoints
        df_intersection[['latitude', 'longitude']] = df_intersection['geopoint'].str.split(',', expand=True)
        df_intersection['latitude'] = df_intersection['latitude'].astype(float)
        df_intersection['longitude'] = df_intersection['longitude'].astype(float)

        # Filtrer par code postal
        filtered_df = df_intersection[df_intersection['code_postal_ban'] == selected_code_postal]

        # Générer la carte
        map_html = generate_map(filtered_df)

        return html.Iframe(srcDoc=map_html, width="100%", height="500px")
