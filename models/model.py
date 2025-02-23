from abc import ABC, abstractmethod
import tensorflow as tf
import os

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