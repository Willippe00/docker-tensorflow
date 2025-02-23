import tensorflow as tf
import numpy as np
import argparse
import os

from train import train
from predict import predict

# Initialisation du parser pour lire le mode de fonctionnement
parser = argparse.ArgumentParser(description="Script TensorFlow avec modes")
parser.add_argument("mode", choices=["train", "predict"], help="Mode de fonctionnement : train ou predict")
args = parser.parse_args()

# Définition du chemin du modèle
vp_path = "/app/vp"
model_path = os.path.join(vp_path, "model.h5")

# Vérifier si le répertoire existe, sinon le créer
os.makedirs(vp_path, exist_ok=True)

if args.mode == "train":
    print("debug log")
    train(model_path)

elif args.mode == "predict":
    predict(model_path)
    
