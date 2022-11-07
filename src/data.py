# Created by A. MATHIEU at 14/10/2022
import pandas as pd
import numpy as np
import pickle
import os

from tqdm import tqdm
from pvlib.solarposition import get_solarposition
from pvlib.temperature import pvsyst_cell

from src.config import ROOT


def load_sample_data():
    """ Load sample data for testing purposes"""

    start = "20190801"
    end = "20191101"
    freq = "15min"
    index = pd.date_range(start, end, freq=freq, tz="CET")

    # Load satellite weather data: incident irradiation, direct irradiation, temperature, wind
    weather_data = pd.read_csv(ROOT / "data" / "meteo_sample.csv", index_col=0)
    weather_data["temp_air"] = weather_data["temp_air"] - 273.15
    weather_data.index = pd.to_datetime(weather_data.index)

    # Get ppm data from Grenoble as example
    pm = get_pm_data("20210801", "20211101", freq=freq).tz_convert("UTC")
    pm.index = (pm.index - pd.DateOffset(years=2)).tz_convert("CET")
    weather_data["pm_2_5_g.m3"] = pm["pm_2_5_g.m3"]
    weather_data["pm_10_g.m3"] = pm["pm_10_g.m3"]
    weather_data = weather_data.reindex(index)

    # Load  pv data
    pv_data = pd.read_csv(ROOT / "data" / "pv_data_sample.csv", index_col=0)  # Vdc, Idc, Pac
    pv_data["Pdc"] = pv_data["Vdc"] * pv_data["Idc"]  # Calculates DC power
    pv_data.index = pd.to_datetime(pv_data.index)
    pv_data = pv_data.reindex(index)

    # Secret file
    secret = pd.read_csv(ROOT / "data" / "secret.csv", header=None, index_col=0)[1]
    secret.loc[["latitude", "longitude", "tilt", "n_module", "n_diode"]] = \
        secret.loc[["latitude", "longitude", "tilt", "n_module", "n_diode"]].astype(float)

    # Get the sun's path azimuth and elevation
    latitude, longitude = float(secret.loc["latitude"]), float(secret.loc["longitude"])
    solar_pos = get_solarposition(index, latitude, longitude)
    azimuth = solar_pos["azimuth"]
    elevation = solar_pos["elevation"]  # or apparent_elevation ?

    # Roughly estimate the cell temperature
    temp_cell = pvsyst_cell(weather_data["poa_global"], weather_data["temp_air"], weather_data["wind_speed"])

    return weather_data, pv_data, azimuth, elevation, temp_cell, secret


def get_pm_data(start_date="20210101",
                end_date="20210110",
                freq="H",
                site: str = "Grenoble Les Frenes",
                store_pkl: bool = True) -> pd.DataFrame:
    """

    Extract airborne particulate matter (PM) concentration with aerodynamic diameter less than 2.5 and 10 microns from
    data.gouv.
    https://files.data.gouv.fr/lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel

    Data available only from 2021 for Grenoble.
    Hourly granularity.

    :param start_date: Fate from which to start to extract particule matter concentrations
    :param end_date: Fate to which to stop to extract particule matter concentrations
    :param freq: Frequency of the recipient (forward fill will be applied if the frequency if finer than an hour)
    :param site: Which site to extract data for
    :param store_pkl: If true, store the data into pickle file
    :return: Dataframe with Datetime Index containing PM2.5 and PM10
    """
    dt_range = pd.date_range(start_date, end_date, freq=freq, tz="CET", inclusive="left")

    date_range_str = dt_range.min().strftime("%Y_%m_%d_%H%M") + "_" + \
                     dt_range.max().strftime("%Y_%m_%d_%H%M") + "_" + \
                     dt_range.freqstr

    path_store = str(ROOT / "data" / "pkls" / f"pm_data_{site.replace(' ', '_')}_{date_range_str}.pkl")
    if store_pkl and os.path.exists(path_store):
        with open(path_store, "rb") as input_file:
            pm_data = pickle.load(input_file)

    else:
        pm_data = pd.DataFrame(index=dt_range, columns=["pm_2_5_g.m3", "pm_10_g.m3"])

        for date in tqdm(np.unique(dt_range.date)):
            year = date.strftime("%Y")
            date_str = date.strftime("%Y-%m-%d")
            url = f"https://files.data.gouv.fr/lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel/" \
                  f"{year}/FR_E2_{date_str}.csv"
            raw_data = pd.read_csv(url, sep=";")
            raw_data_site = raw_data[(raw_data["nom site"] == site)]

            data_10 = raw_data_site[(raw_data_site["Polluant"] == "PM10")].set_index("Date de début")["valeur"]
            data_25 = raw_data_site[(raw_data_site["Polluant"] == "PM2.5")].set_index("Date de début")["valeur"]
            data_10.index = pd.to_datetime(data_10.index).tz_localize("CET", ambiguous=True,
                                                                      nonexistent='shift_forward')
            data_25.index = pd.to_datetime(data_25.index).tz_localize("CET", ambiguous=True,
                                                                      nonexistent='shift_forward')
            pm_data.loc[data_10.index, "pm_10_g.m3"] = data_10 / 1000 / 1000  # conversion en g/m3
            pm_data.loc[data_25.index, "pm_2_5_g.m3"] = data_25 / 1000 / 1000  # conversion en g/m3

        pm_data = pm_data.ffill()

        if store_pkl and os.path.exists(str(ROOT / "data" / "pkls")):
            with open(path_store, 'wb') as handle:
                pickle.dump(pm_data, handle)

    return pm_data


if __name__ == '__main__':
    weather_data, pv_data, azimuth, elevation, temp_cell, secret = load_sample_data()
