import csv
import os
from datetime import datetime
from enum import Enum

from herbie import Herbie, FastHerbie
import xarray as xr
import numpy as np
import pandas as pd

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

    def getMeteoEntrainement(self, annee, mois, jour, stations, intrants):

        date_str = f"{annee:04d}-{mois:02d}-{jour:02d}"

        output = []
        for row in self.data:
            
            if row[0] == date_str:

                for intrant in intrants:

                    col_index = self.headers.index(self.conversionNomColone(intrant))
                    output.append(row[col_index])
                return output

        return np.zeros(len(intrants), dtype=int)
    
    def getMeteoPrediction(self, annee, mois, jour, heure,stations, intrants):
        
        lat_mtl = 45.5017
        lon_mtl = -73.5673

        heure = 0 # la prévision du matin
        
        run_date = datetime(annee, mois, jour, heure).strftime("%Y-%m-%d %H:%M") # changer pour le matin

        model = "hrrr"

        variable = "TMP:2 m above ground"

        ######develop##############
        temperature_list = []

        for fxx in range(0, 25, 1):
            H = Herbie(run_date, model=model, product="sfc", fxx=fxx)
            ds = H.xarray(variable, remove_grib=False)

            distance = (ds.latitude - lat_mtl) ** 2 + (ds.longitude - lon_mtl) ** 2
            y_idx, x_idx = np.unravel_index(distance.argmin(), ds.latitude.shape)

            temp_k = ds.t2m.isel(y=y_idx, x=x_idx).values.item()
            print("température kelvin")
            print(temp_k)
            temp_c = temp_k - 273.15
            temperature_list.append( temp_c)
        ###########################

        print("liste final")
        print(temperature_list)

        return [max(temperature_list),min(temperature_list) ]
    
    

    
    def conversionNomColone(self, var : MeteoVar):
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