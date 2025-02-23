from datetime import datetime
import tensorflow as tf
import numpy as np

from models.model import BaseModel, ModelVar

from intrant.intrantMW import getDonneHistoriqueOuvert
from intrant.intrantMeteo import intrantMeteo, MeteoVar

class ModelW(BaseModel):

    def __init__(self):
        #super().__init__(model_path)
        self.model = None
        self.epochs = 30

        self.intrantsModel = [ModelVar.MOIS, ModelVar.JOUR ,ModelVar.HEURE]
        self.intrantsMeteo = [MeteoVar.MAXTEMPDAY, MeteoVar.MINTEMPDAY]


        self.input_shape = self.ComputeImputShap([self.intrantsModel, self.intrantsMeteo]) #a modifier selon l'architecture
        self.model = self.build_model()

    def build_model(self): # a modfier

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=self.input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, X_train, y_train, batch_size=16):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_input):
        """Effectue une prédiction"""
        X_input = np.array(X_input).reshape(-1, *self.input_shape)  # Adapter la forme
        return self.model.predict(X_input)
    

    def getDataEntraiment(self):

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

            X_row.extend([mois, jour, heure])

            VarsMeteo = intantMeteo.getMeteo(annee, mois, jour , "MTL", self.intrantsMeteo)
            X_row.extend(VarsMeteo)
            X_train.append(X_row)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        return X_train, y_train