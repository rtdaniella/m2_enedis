# Appeler la fonction d'entraînement pour le modèle de classification
from classifier_model import train_classifier
from regressor_model import train_regressor


print("Entraînement du modèle de classification...")
train_classifier()  # Cette fonction entraînera le modèle de classification et le sauvegardera

# Appeler la fonction d'entraînement pour le modèle de régression
# print("Entraînement du modèle de régression...")
# train_regressor()  # Cette fonction entraînera le modèle de régression et le sauvegardera

print("Les deux modèles ont été entraînés avec succès.")