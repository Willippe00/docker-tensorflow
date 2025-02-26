from datetime import datetime
import tensorflow as tf
import numpy as np

from models.model import BaseModel, ModelVar

from intrant.intrantMW import getDonneHistoriqueOuvert
from intrant.intrantMeteo import intrantMeteo, MeteoVar

class ModelW(BaseModel):

    def __init__(self):
        super().__init__()

        self.nomModel = "ModelW-1"
        self.model = None
        self.epochs = 20
        self.batch_size = 16

        self.intrantsModel = [ModelVar.MOIS, ModelVar.JOUR ,ModelVar.HEURE, ModelVar.WEEKDAY]
        self.intrantsMeteo = [MeteoVar.MAXTEMPDAY, MeteoVar.MINTEMPDAY, MeteoVar.MAXWINDSPEEDDAY]


        self.input_shape = self.ComputeImputShap([self.intrantsModel, self.intrantsMeteo]) #a modifier selon l'architecture
        self.model = self.build_model()

    def build_model(self): # a modfier

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=self.input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["mae"])
        return model

    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size= self.batch_size, verbose=1)

    def predict(self, annee, mois, jour, heure):
        
       return self.predictHerbie(annee, mois, jour, heure)
    

    def getDataEntraiment(self):

        return self.getDataEntraimentHistoriqueOuvert()