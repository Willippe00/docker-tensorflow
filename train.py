import tensorflow as tf
import numpy as np
import argparse
import os

def train(model_path):
    print("Entraînement du modèle...")

    # Données factices pour l'entraînement
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    # Création du modèle simple
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Entraînement du modèle
    model.fit(X, y, epochs=5, verbose=1)

    # Sauvegarde du modèle
    model.save(model_path)
    print("dans fichier externe")
    print(f"Modèle sauvegardé à : {model_path}")
    