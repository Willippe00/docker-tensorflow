import tensorflow as tf
import numpy as np
import argparse
import os

def predictManuel(model_path, models):

    print("Mode prédiction activé")

    
    annee = 2025
    mois = 2
    jour = 23
    heure = 17

    for model in models:
        model.load_model_latest(model_path)
        prediction = model.predict(annee, mois, jour, heure)

        
        #print(f"Prédiction pour {sample}: {prediction}")
