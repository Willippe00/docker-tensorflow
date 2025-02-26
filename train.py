import tensorflow as tf
import numpy as np
import argparse
import os

def trainManuel(model_path, models):
    print("Entraînement du modèle...")


    for model in models:
        X_train, y_train = model.getDataEntraiment()
        model.train(X_train, y_train)
        model.save_model(model_path)
