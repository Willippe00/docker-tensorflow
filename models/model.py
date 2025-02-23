from abc import ABC, abstractmethod
import tensorflow as tf
import os
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
    def predict(self, X_input):
        """Effectue une prédiction avec le modèle entraîné."""
        pass

    @abstractmethod
    def getDataEntraiment(self):
        """Effectue une prédiction avec le modèle entraîné."""
        pass


    def save_model(self, model_path):
        """Sauvegarde le modèle en fichier .h5"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Modèle sauvegardé à : {model_path}")
        else:
            print("Erreur : Aucun modèle à sauvegarder.")

    def load_model(self, model_path):
        """Charge un modèle à partir d'un fichier .h5"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
            print(f"Modèle chargé depuis : {model_path}")
        else:
            print("Erreur : Aucun modèle trouvé à charger.")

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



