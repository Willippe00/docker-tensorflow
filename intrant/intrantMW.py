import csv
import os

fileHistoOuvertAgreger = os.path.join("/app/data/agreger", "data.csv")#R".\data\agreger\data.csv"

def getDonneHistoriqueOuvert():
    """out au format dictionaire "2021-12-31  08:00:00" -> MW"""


    data_dict = {}

    with open(fileHistoOuvertAgreger, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")  # Lire ligne par ligne
        for row in reader:
            if len(row) == 2:  # Vérifier qu'il y a bien 2 colonnes
                key, value = row
                if value.strip() == "":  # Vérifier si la valeur est vide
                    print(f"⚠️ Valeur vide détectée pour la clé : {key}")
                    #data_dict[key] = -1 #a retirer éventuelment
                    continue  # Passer à la prochaine clé

                data_dict[key] = float(value.replace(",", "."))
    return data_dict