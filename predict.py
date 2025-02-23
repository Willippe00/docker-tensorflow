import tensorflow as tf
import numpy as np
import argparse
import os

def predict(model_path):

    print("Mode prédiction activé")

    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print("Erreur : Le modèle n'existe pas. Entraîne d'abord le modèle avec 'train'.")
        exit(1)

    # Charger le modèle
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès.")

    # Données factices pour la prédiction
    sample = np.array([[7, 8]])
    prediction = model.predict(sample)

    print(f"Prédiction pour {sample}: {prediction}")