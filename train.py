import tensorflow as tf
import numpy as np
import argparse
import os

def train(model_path, models):
    print("Entraînement du modèle...")


    #####jeu de donné bidon#####
    X_train = np.random.rand(100, 2)
    y_train = np.random.randint(0, 2, size=(100,))  # Classification binaire (0 ou 1)
    ############################

  
    for model in models:
        model.train(X_train, y_train, batch_size=32)
        model.save_model(model_path)


    print("dans fichier externe")

    