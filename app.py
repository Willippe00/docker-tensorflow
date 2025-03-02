import os

import tensorflow as tf
import numpy as np
import argparse

from dash_app import dash_app


from train import trainManuel
from predict import predictManuel
from accuracy import accuracy


from models.W.modelW import ModelW

# Initialisation du parser pour lire le mode de fonctionnement
parser = argparse.ArgumentParser(description="Script TensorFlow avec modes")
parser.add_argument("mode", choices=["train", "predict", "accuracy", "dash"], help="Mode de fonctionnement : train ou predict")
args = parser.parse_args()

# Définition du chemin du modèle
base_path = "/app/vp"
#model_path = os.path.join(vp_path, "model.h5")

# Vérifier si le répertoire existe, sinon le créer
os.makedirs(base_path, exist_ok=True)

print("lancment du conteneur de prévision")

models = [ModelW()]

if args.mode == "train":
    print("debug log")
    trainManuel(base_path, models)

elif args.mode == "predict":
    predictManuel(base_path, models)

elif args.mode == "accuracy":
    accuracy(base_path, models, nbStep=2)

elif args.mode == "dash":
    dash_app.app.run(debug=True, host="0.0.0.0", port=8050)

    
