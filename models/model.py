from abc import ABC, abstractmethod
import tensorflow as tf
import os
import shutil
from enum import Enum
from datetime import datetime
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
import re


from intrant.intrantMW import getDonneHistoriqueOuvert
from intrant.intrantMeteo import intrantMeteo, MeteoVar

class ModelVar(Enum):
    ANNEE = "annee"
    MOIS = "mois"
    JOUR = "jour"
    HEURE = "heure"
    WEEKDAY = "weekday"

class BaseModel(ABC):

    def __init__(self):
        self.nomModel = "BaseModel"
        self.model = None
        self.epochs = 10
        self.batch_size = 8

        self.optimizer = "adam"
        self.loss = "mse"


    @abstractmethod
    def build_model(self):
        """Doit créer et retourner un modèle Keras."""
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        """Entraîne le modèle avec les données fournies."""
        pass

    @abstractmethod
    def predict(self, annee, mois, jour, heure):
        """Effectue une prédiction avec le modèle entraîné."""
        pass

    @abstractmethod
    def getDataEntraiment(self):
        """Effectue une prédiction avec le modèle entraîné."""
        pass

    def predictHerbie(self, annee, mois, jour, heure):
        """Effectue une prédiction"""
        
        X_input = []

        X_row = []
       
        X_row.extend(self.getValueImputModel(annee=annee,mois=mois,jour=jour,heure=heure)) # a modifer avec paramètre

        intantMeteo = intrantMeteo()
        X_row.extend(intantMeteo.getMeteoPrediction(annee=annee,mois=mois,jour=jour,heure=heure,stations="mtl", intrants=self.intrantsMeteo))

        print("X_row!!")
        print(X_row)
        X_input.append(X_row)
        X_input = np.array(X_input)

        print(X_input)


        prediction =  self.model.predict(X_input)
        print(f"Prédiction pour {X_input}: {prediction}")

        return prediction

    def getDataEntraimentHistoriqueOuvert(self):
        data_dict = getDonneHistoriqueOuvert()

        intantMeteo = intrantMeteo()
        
        X_train = []
        y_train = []


        for key, value in data_dict.items():
            X_row = []
            y_train.append(value)

            time_obj = datetime.strptime(key, "%Y-%m-%d %H:%M")

            annee = time_obj.year
            mois = time_obj.month
            jour = time_obj.day
            heure = time_obj.hour
            minute = time_obj.minute
            weekday = time_obj.isoweekday()

            X_row.extend([mois, jour, heure, weekday])

            VarsMeteo = intantMeteo.getMeteoEntrainement(annee, mois, jour , "MTL", self.intrantsMeteo)
            X_row.extend(VarsMeteo)
            X_train.append(X_row)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        return X_train, y_train


    def save_model(self, base_path, model_name="model.h5"):
        """
        Sauvegarde le modèle dans :
        1. "data_latest/model.h5" (dernière version)
        2. "YYYY-MM-DD_HH-MM/model.h5" (version historisée)
        """

        # Chemin du dossier "camion"
        camion_path = os.path.join(base_path, self.nomModel)

        # Vérifier et créer "camion" si nécessaire
        if not os.path.exists(camion_path):
            os.makedirs(camion_path)
            print(f'📂 Dossier créé: {camion_path}')
        else:
            print(f'✅ Dossier existe: {camion_path}')

        # Chemin du dossier "data_latest"
        data_latest_path = os.path.join(camion_path, "data_latest")

        # Supprimer et recréer "data_latest"
        if os.path.exists(data_latest_path):
            shutil.rmtree(data_latest_path)
            print(f'🗑️ Dossier "data_latest" supprimé.')

        os.makedirs(data_latest_path)
        print(f'📂 Dossier "data_latest" recréé.')

        # Générer le dossier horodaté au même niveau que "data_latest"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        timestamp_folder = os.path.join(camion_path, timestamp)  # 🔹 Dossier au même niveau que "data_latest"
        os.makedirs(timestamp_folder)
        print(f'📂 Dossier horodaté créé: {timestamp_folder}')

        # Définir les chemins pour la sauvegarde
        model_path_latest = os.path.join(data_latest_path, model_name)
        model_path_timestamped = os.path.join(timestamp_folder, model_name)





        # Sauvegarder le modèle dans "data_latest" et dans le dossier horodaté
        self.model.save(model_path_latest, save_format="h5")
        print(f'✅ Modèle sauvegardé dans "data_latest": {model_path_latest}')

        self.model.save(model_path_timestamped, save_format="h5")
        print(f'✅ Modèle sauvegardé dans le dossier horodaté: {model_path_timestamped}')

        pedigree_path_latest  = os.path.join(data_latest_path, f"pedigree_{timestamp}.json")
        pedigree_path_timestamped  = os.path.join(timestamp_folder, f"pedigree_{timestamp}.json")

        pedigree = {
            "intrantsModel": [var.value for var in self.intrantsModel],#self.intrantsModel, 
            "intrantsMeteo": [var.value for var in self.intrantsMeteo],#self.intrantsMeteo, 
            "epochs": self.epochs, 
            "batch_size": self.batch_size,
            "optimizer" : self.optimizer,
            "loss" : self.loss}

        with open(pedigree_path_latest, "w") as f:
            json.dump(pedigree, f, indent=4)

        with open(pedigree_path_timestamped, "w") as f:
            json.dump(pedigree, f, indent=4)


    def load_model_latest(self, base_path, model_name="model.h5"):
        """
        Charge le dernier modèle sauvegardé dans "data_latest", en gérant les fonctions personnalisées.
        """
        model_path = os.path.join(base_path, self.nomModel, "data_latest", model_name)

        if not os.path.exists(model_path):
            print(f'❌ Aucun modèle trouvé dans {model_path}')
            return None

        # Charger avec `custom_objects` pour éviter l'erreur sur `mse`
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
            print(f'✅ Modèle chargé depuis: {model_path}')
            return self.model
        except Exception as e:
            print(f'⚠️ Erreur lors du chargement du modèle: {e}')
            return None

    def saveOutPutModel(self):
        print("sortie Sauvegarder")

    def ComputeImputShap(self, listesTypesIntrant):

        nbIntran = 0
        for TypesIntrant in listesTypesIntrant:
            nbIntran += len(TypesIntrant)
        return (nbIntran,)
    
    def getValueImputModel(self, annee, mois, jour, heure, now = False):
        date_actuelle = datetime.today()

        intrantList = []
        if now:
            for intrant in self.intrantsModel:

                if intrant.value == ModelVar.ANNEE.value:
                    intrantList.append(date_actuelle.year)
                elif  intrant.value == ModelVar.MOIS.value:
                    intrantList.append(date_actuelle.month)
                elif  intrant.value == ModelVar.JOUR.value:
                    intrantList.append(date_actuelle.day)
                elif  intrant.value == ModelVar.HEURE.value:
                    intrantList.append(date_actuelle.hour)
                elif intrant.value == ModelVar.WEEKDAY.value:
                    intrantList.append(date_actuelle.isoweekday())
                else:
                    raise ValueError(f"Intrant inconnu : {intrant.value}. Attendu : {list(ModelVar)}")
        else:
            for intrant in self.intrantsModel:

                if intrant.value == ModelVar.ANNEE.value:
                    intrantList.append(annee)
                elif  intrant.value == ModelVar.MOIS.value:
                    intrantList.append(mois)
                elif  intrant.value == ModelVar.JOUR.value:
                    intrantList.append(jour)
                elif  intrant.value == ModelVar.HEURE.value:
                    intrantList.append(heure)
                elif intrant.value == ModelVar.WEEKDAY.value:
                    intrantList.append(datetime(annee, mois, jour).isoweekday())
                else:
                    raise ValueError(f"Intrant inconnu : {intrant.value}. Attendu : {list(ModelVar)}")
        return intrantList
    
    def evaluate_latest_model(self, base_path, nbStep):
        """
        Évalue le dernier modèle sur les données récentes et enregistre les métriques.
        """
        model = self.load_model_latest(base_path)
        if model is None:
            return
        
        url = f"https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/demande-electricite-quebec/records?limit={nbStep}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as err:
            print(f"⚠️ Erreur API : {err}")
            return
        
        records = data["results"]
        y_true, y_pred = [], []

        for record in records:
            date_obj = datetime.fromisoformat(record["date"])
            annee, mois, jour, heure = date_obj.year, date_obj.month, date_obj.day, date_obj.hour

            prediction = self.predict(annee, mois, jour, heure)
            valeur_reelle = record["valeurs_demandetotal"]  # Change selon ta structure
            
            y_true.append(valeur_reelle)
            y_pred.append(prediction[0][0])

        # 🔹 Calcul des métriques
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 🔹 Affichage des résultats
        print(f"📊 Performance du modèle ({self.nomModel}):")
        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.2f}")

        # Récupérer le timestamp du pedigree
        pedigree_timestamp = self.get_latest_pedigree_timestamp(base_path, self.nomModel)
        evaluation_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # 🔹 Création du dossier de sauvegarde
        performance_folder = os.path.join(base_path, self.nomModel, "performance")
        os.makedirs(performance_folder, exist_ok=True)

        # 🔹 Sauvegarde des métriques dans un fichier JSON avec le timestamp
        performance_path = os.path.join(performance_folder, f"metrics_{evaluation_timestamp}.json")
        metrics = {
            "timestamp": pedigree_timestamp,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        with open(performance_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Métriques sauvegardées : {performance_path}")

    def get_latest_pedigree_timestamp(self, base_path, model_name):
        """
        Récupère le timestamp depuis le nom du fichier `pedigree_YYYY-MM-DD_HH-MM.json` dans `data_latest`.
        """
        data_latest_path = os.path.join(base_path, model_name, "data_latest")

        if not os.path.exists(data_latest_path):
            print(f"❌ Le dossier {data_latest_path} n'existe pas.")
            return datetime.now().strftime("%Y-%m-%d_%H-%M")  # Fallback

        # 🔹 Liste les fichiers dans `data_latest/`
        try:
            files = os.listdir(data_latest_path)
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {data_latest_path}: {e}")
            return datetime.now().strftime("%Y-%m-%d_%H-%M")  # Fallback

        # 🔹 Filtrer les fichiers correspondant à `pedigree_YYYY-MM-DD_HH-MM.json`
        pattern = re.compile(r"pedigree_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.json")

        for file in files:
            match = pattern.match(file)
            if match:
                return match.group(1)  # Retourne le timestamp extrait du nom du fichier

        print("❌ Aucun fichier `pedigree_YYYY-MM-DD_HH-MM.json` trouvé dans data_latest.")
        return datetime.now().strftime("%Y-%m-%d_%H-%M")  # Fallback si rien n'est trouvé







