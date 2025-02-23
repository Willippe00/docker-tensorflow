# Utiliser une image officielle avec TensorFlow préinstallé
FROM tensorflow/tensorflow:latest

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY app.py .
COPY train.py .
COPY predict.py .

#copier les données pour l'entrainement
COPY data /app/data

#stocker les h5
#VOLUME ["/app/vp"]
RUN mkdir -p /app/vp

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande de démarrage
ENTRYPOINT  ["python", "app.py"]
