from abc import ABC, abstractmethod
import tensorflow as tf
import os
import shutil
from enum import Enum
from datetime import datetime

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


    @abstractmethod
    def build_model(self):
        """Doit créer et retourner un modèle Keras."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, batch_size=32):
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



