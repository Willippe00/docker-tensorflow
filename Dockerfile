# Utiliser une image officielle avec TensorFlow préinstallé
FROM tensorflow/tensorflow:latest

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY app.py .
COPY train.py .
COPY predict.py .
COPY models/model.py ./models/model.py
COPY models/W/modelW.py ./models/W/modelW.py

COPY intrant/intrantMW.py ./intrant/intrantMW.py

#copier les données pour l'entrainement
COPY data /app/data

RUN mkdir -p ./data/agreger
COPY data/agreger/data.csv ./data/agreger/data.csv

#COPY data/agreger/data.csv ./data/agreger/data.csv


#stocker les h5
#VOLUME ["/app/vp"]
RUN mkdir -p /app/vp

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir la commande de démarrage
ENTRYPOINT  ["python", "app.py"]
