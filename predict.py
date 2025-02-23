import tensorflow as tf
import numpy as np
import argparse
import os

def predict(model_path, models):

    print("Mode prédiction activé")

    
    sample = np.array([[2, 23, 11, -3, -13]])
    

    for model in models:
        model.load_model(model_path)
        prediction = model.predict(sample)

        
        print(f"Prédiction pour {sample}: {prediction}")
