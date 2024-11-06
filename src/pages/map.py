from dash import html
import pandas as pd
import folium
from folium.plugins import MarkerCluster


def create_map_page(X_train,X_test,y_train,y_test,df_all):
    df_all = pd.DataFrame(df_all).set_index('n_dpe')
# Extraire les geopoints pour X_train et X_test à partir de df_all
    X_train['geopoint'] = df_all.loc[X_train.index, 'geopoint']
    X_test['geopoint'] = df_all.loc[X_test.index, 'geopoint']

# Combiner les DataFrames X_train et X_test avec y_train et y_test
    combined_df = pd.concat([X_train, y_train], axis=1)
    combined_df = combined_df.rename(columns={0: 'target'})  # Renommer la colonne de y_train

    combined_test_df = pd.concat([X_test, y_test], axis=1)
    combined_test_df = combined_test_df.rename(columns={0: 'target'})  # Renommer la colonne de y_test

# Combiner les ensembles d'entraînement et de test
    final_df = pd.concat([combined_df, combined_test_df])
    # Séparer les coordonnées de geopoint
    final_df[['latitude', 'longitude']] = final_df['geopoint'].str.split(',', expand=True)
    final_df['latitude'] = final_df['latitude'].astype(float)
    final_df['longitude'] = final_df['longitude'].astype(float)
# Filtrer le DataFrame pour n'inclure que les logements avec le code postal 69001
    filtered_df = final_df[final_df['code_postal_ban'] == "69001"]

# Créer une carte centrée sur la moyenne des coordonnées des logements filtrés
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

# Créer un cluster pour les marqueurs
    marker_cluster = MarkerCluster().add_to(m)

# Ajouter les points de géopositionnement à la carte pour le DataFrame filtré
    for _, row in filtered_df.iterrows():
        folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['type_batiment']} - {row['logement']}"
        ).add_to(marker_cluster)
    

    return html.Div([
        html.H1("Page Map"),
        html.P("Ceci est le contenu de la page Map."),
        html.Div([m])
    ])
