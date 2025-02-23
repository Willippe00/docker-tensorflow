from datetime import datetime
import tensorflow as tf
import numpy as np

from models.model import BaseModel
from intrant.intrantMW import getDonneHistoriqueOuvert

class ModelW(BaseModel):

    def __init__(self):
        #super().__init__(model_path)
        self.model = None
        self.epochs = 40   
        self.input_shape = (3,) #a modifier selon l'architecture
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
        #X_input = np.array(X_input).reshape(-1, *self.input_shape)  # Adapter la forme
        return self.model.predict(X_input)
    
    def getDataEntraiment(self):
        # #####jeu de donné bidon#####
        # X_train = np.random.rand(100, 2)
        # y_train = np.random.randint(0, 2, size=(100,))  # Classification binaire (0 ou 1)
        # ############################


        
        # y_train = np.array([v for v in data_dict.values()])[:100]

        # #y_train = np.array()

        data_dict = getDonneHistoriqueOuvert()

        X_train = []
        y_train = []

        for key, value in data_dict.items():
            y_train.append(value)
            #print(value)

            time_obj = datetime.strptime(key, "%Y-%m-%d %H:%M")

            mois = time_obj.month
            jour = time_obj.day
            heure = time_obj.hour
            minute = time_obj.minute

            X_train.append([mois, jour, heure])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        


        
        
        return X_train, y_train