import csv
import os
from datetime import datetime, timedelta
import pytz
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from herbie import Herbie, FastHerbie
import xarray as xr
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun

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
    
    def getMeteoPrediction(self, annee, mois, jour, heure, stations, intrants):
        
        lat_mtl = 45.5017
        lon_mtl = -73.5673

        date_cible = datetime(annee, mois, jour)

        heure = 0 # la prévision du matin

        montreal_tz = pytz.timezone("America/Toronto")
        local_datetime = datetime(annee, mois, jour, heure, tzinfo=montreal_tz)
        run_date_utc = local_datetime.astimezone(pytz.utc)

        ville = LocationInfo("Montréal", "Canada", "America/Toronto", lat_mtl, lon_mtl)
        
        run_date = run_date_utc.strftime("%Y-%m-%d %H:%M") # changer pour le matin

        dataList  = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_temperature = executor.submit(self.getListeTemperature, run_date, lon_mtl, lat_mtl)
            future_vvent = executor.submit(self.getListeVent, run_date, lon_mtl, lat_mtl)
            future_precipitation = executor.submit(self.getListePrecipitations, run_date, lon_mtl, lat_mtl)

            # 🔄 Récupérer les résultats une fois prêts
            temperature_list = future_temperature.result()
            VVent_list = future_vvent.result()
            precipitation_list = future_precipitation.result()

        for intrant in intrants:
            if intrant.value == MeteoVar.MAXTEMPDAY.value:
                dataList.append(max(temperature_list))
            elif intrant.value == MeteoVar.MINTEMPDAY.value:
                dataList.append(min(temperature_list))
            elif intrant.value == MeteoVar.MAXWINDSPEEDDAY.value:
                dataList.append(max(VVent_list))
            elif intrant.value == MeteoVar.TOTPRECIPMM.value:
                dataList.append(sum(precipitation_list))
            elif intrant.value == MeteoVar.SUNSET.value:
                dataList.append(sun(ville.observer, date=date_cible)["sunset"].strftime("%H:%M:%S"))
            elif intrant.value == MeteoVar.SUNSET.value:
                dataList.append(sun(ville.observer, date=date_cible)["sunrise"].strftime("%H:%M:%S"))

        return dataList
    

    def getListeTemperature(self, run_date, lon, lat):
        model = "hrrr"

        variable = "TMP:2 m above ground"

        temperature_list = []

        for fxx in range(0, 17, 1): # a ajuster mais juste 18 step
            H = Herbie(run_date, model=model, product="sfc", fxx=fxx)
            ds = H.xarray(variable, remove_grib=False, backend_kwargs={"decode_timedelta": False})

            distance = (ds.latitude - lat) ** 2 + (ds.longitude - lon) ** 2
            y_idx, x_idx = np.unravel_index(distance.argmin(), ds.latitude.shape)

            temp_k = ds.t2m.isel(y=y_idx, x=x_idx).values.item()
            temp_c = temp_k - 273.15
            temperature_list.append( temp_c)

        return temperature_list
    
    def getListeVent(self, run_date, lon, lat):
        model = "hrrr"

        # Variables du vent à 10m
        variable_u = "UGRD:10 m above ground"  # Vent U à 10m
        variable_v = "VGRD:10 m above ground"  # Vent V à 10m

        ###### Développement ##############
        wind_speed_list = []

        for fxx in range(0, 17, 1):  # 18 étapes de prévision
            H = Herbie(run_date, model=model, product="sfc", fxx=fxx)

            ds_u = H.xarray(variable_u, backend_kwargs={"decode_timedelta": False})
            ds_v = H.xarray(variable_v, backend_kwargs={ "decode_timedelta": False})

            distance = (ds_u.latitude - lat) ** 2 + (ds_u.longitude - lon) ** 2
            y_idx, x_idx = np.unravel_index(distance.argmin(), ds_u.latitude.shape)

            u10 = ds_u.u10.isel(y=y_idx, x=x_idx).values.item()
            v10 = ds_v.v10.isel(y=y_idx, x=x_idx).values.item()

            wind_speed = np.sqrt(u10**2 + v10**2)
            wind_speed_list.append(wind_speed)

        return wind_speed_list
    
    def getListePrecipitations(self, run_date, lon, lat):
        model = "hrrr"
        variable = "APCP:surface"  # Précipitations accumulées

        precipitation_list = []
        previous_precip = 0  # Stocker la valeur précédente pour calculer les précipitations horaires

        for fxx in range(0, 17, 1):  # 18 étapes de prévision
            H = Herbie(run_date, model=model, product="sfc", fxx=fxx)

            ds = H.xarray(variable, remove_grib=False, backend_kwargs={ "decode_timedelta": False})

            distance = (ds.latitude - lat) ** 2 + (ds.longitude - lon) ** 2
            y_idx, x_idx = np.unravel_index(distance.argmin(), ds.latitude.shape)


            precip_accumulated = ds.tp.isel(y=y_idx, x=x_idx).values.item()

            precip_hourly = precip_accumulated - previous_precip
            precipitation_list.append(max(precip_hourly, 0))  # Éviter les valeurs négatives

            previous_precip = precip_accumulated

        return precipitation_list
    
    

    
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
        elif var.value == MeteoVar.SUNRISE.value:
            return "SUNRISE"