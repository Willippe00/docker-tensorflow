import tensorflow as tf
import numpy as np
import argparse
import os

def predict(model_path, models):

    print("Mode prédiction activé")

    
    sample = np.array([[5, 15, 10]])
    

    for model in models:
        model.load_model(model_path)
        prediction = model.predict(sample)

        
        print(f"Prédiction pour {sample}: {prediction}")
