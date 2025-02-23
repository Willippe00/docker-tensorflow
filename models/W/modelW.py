from models.model import BaseModel
import tensorflow as tf
import numpy as np

class ModelW(BaseModel):

    def __init__(self):
        #super().__init__(model_path)
        self.model = None
        self.epochs = 40   
        self.input_shape = (2,) #a modifier selon l'architecture
        self.model = self.build_model()

    def build_model(self): # a modfier

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=self.input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, X_train, y_train, batch_size=32):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_input):
        """Effectue une prédiction"""
        X_input = np.array(X_input).reshape(-1, *self.input_shape)  # Adapter la forme
        return self.model.predict(X_input)