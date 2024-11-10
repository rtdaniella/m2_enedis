Utilisation des Modèles Joblib
==============================

Les modèles entraînés sont sauvegardés dans des fichiers `.joblib` pour une utilisation ultérieure. Voici comment les charger et les utiliser.

Charger un modèle
-----------------
Utilisez le module `joblib` pour charger les modèles :

.. code-block:: python

    from joblib import load

    # Remplacez 'path/to/model.joblib' par le chemin vers votre modèle
    model = load('src.utils.models.classifier_model_v0.joblib')

Utiliser le modèle
------------------
Après avoir chargé le modèle, vous pouvez l'utiliser pour faire des prédictions :

.. code-block:: python

    # Exemple de prédiction
    prediction = model.predict(data)
    print("La prédiction du modèle est :", prediction)
