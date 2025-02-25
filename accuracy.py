import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime

import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def accuracy(model_path, models, nbStep):


    url = "https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/demande-electricite-quebec/records?limit="

    urlfinal = url + str(nbStep)

    response = requests.get(urlfinal)

    
    try:
        response.raise_for_status()  # Lève une exception si le statut n'est pas 200
        data = response.json()
    except requests.exceptions.HTTPError as errh:
        raise Exception(f"Erreur HTTP {response.status_code}: {errh}")
    except requests.exceptions.RequestException as err:
        raise Exception(f"Erreur de connexion à l'API : {err}")
    
    records  = data["results"]

    

    annee = 2025
    mois = 2
    jour = 23
    heure = 17

    

    for model in models:
        model.load_model_latest(model_path)

        y_true = []
        y_pred = []

        for record in records:

            date_obj = datetime.fromisoformat(record["date"])

            # Extraire les éléments
            annee = date_obj.year
            mois = date_obj.month
            jour = date_obj.day
            heure = date_obj.hour

            prediction = model.predict(annee, mois, jour, heure)

            # Récupérer la valeur réelle (ajuste selon ton dataset)
            valeur_reelle = record["valeurs_demandetotal"]  # Change "valeur_reelle" selon ta structure

            # Stocker les valeurs
            y_true.append(valeur_reelle)
            y_pred.append(prediction[0][0]) # a modifier 

        # 🔹 Calcul des métriques (Régression)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 🔹 Affichage des résultats
        print(f"📊 Performance du modèle :")
        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.2f}")