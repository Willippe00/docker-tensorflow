import csv
import os
from datetime import datetime
from enum import Enum

import numpy as np

fileHistoOuvertAgreger = os.path.join("/app/data/agreger", "meteo.csv")

class MeteoVar(Enum):
    ###################max a day#################
    MAXTEMPDAY = "MaxTDay"
    MINTEMPDAY = "MinTDay"

    MAXWINDSPEEDDAY = "MaxVDay"

    TOTPRECIPMM = "TotPrecipMM"

    SUNSET  = "sunset"
    SUNRISE = "sunrise"
    ############################################


    TEMPERATURE = "Température"
    VENT = "Vent"
    PRECIPITATION = "Précipitation"

class intrantMeteo():

    def __init__(self):
        with open(fileHistoOuvertAgreger, mode="r", encoding="utf-8") as file:
            self.reader = list(csv.reader(file, delimiter=","))
            self.headers = self.reader[0]
            self.data = self.reader[1:]

    def getMeteo(self, annee, mois, jour, stations, intrants):

        date_str = f"{annee:04d}-{mois:02d}-{jour:02d}"

        output = []
        for row in self.data:
            
            if row[0] == date_str:

                for intrant in intrants:

                    col_index = self.headers.index(self.conversionNomColone(intrant))
                    output.append(row[col_index])
                return output

        return np.zeros(len(intrants), dtype=int)
    
    def conversionNomColone(var : MeteoVar):
        if var.value == MeteoVar.MAXTEMPDAY.value:
            return "MAX_TEMPERATURE_C"
        elif var.value == MeteoVar.MINTEMPDAY.value:
            return "MIN_TEMPERATURE_C"
        elif var.value == MeteoVar.MAXWINDSPEEDDAY.value:
            return "WINDSPEED_MAX_KMH"
        elif var.value == MeteoVar.TOTPRECIPMM.value:
            return "PRECIP_TOTAL_DAY_MM"
        elif var.value == MeteoVar.SUNSET.value:
            return "SUNSET"
        elif var.value == MeteoVar.SUNSET.value:
            return "SUNRISE"