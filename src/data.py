# Created by A. MATHIEU at 14/10/2022
import pandas as pd
import numpy as np
import json
import pickle
import os

from tqdm import tqdm
from pathlib import Path

from src.config import ROOT


def import_config(config_path: Path) -> dict:
    """Import parameters from a json file"""
    with open(config_path) as config_file:
        config = json.load(config_file)
    missing_keys = {'site', 'analysis', 'irr', 'therm', 'techno', 'layout'} - set(config.keys())
    if len(missing_keys) > 0:
        pass
    # Parse additional data:
    if 'additional_data' in config['site'].keys():
        for key, val in config['site']['additional_data'].items():
            config['site'][key] = val
        del config['site']['additional_data']

    return config


def import_pv_data(pv_path_list) -> pd.DataFrame:
    """Import pv data from a json file"""
    pv_data = []
    for pv_path in pv_path_list:
        pv_data.append(pd.read_json(pv_path, orient="index"))
    if len(pv_data) > 0:
        pv_data = pd.concat(pv_data, axis=0)
    else:
        pv_data = pd.DataFrame()
    if not pv_data.index.is_monotonic_increasing:
        pass
    return pv_data


def import_weather_data(meteo_path_list) -> pd.DataFrame:
    """Import weather data from a json file"""
    meteo_data = []
    for meteo_path in meteo_path_list:
        with open(meteo_path) as meteo_file:
            meteo_json = json.load(meteo_file)
        meteo_data.append(pd.read_json(json.dumps(meteo_json['data']), orient="index"))
    meteo_data = pd.concat(meteo_data, axis=0)
    if not meteo_data.index.is_monotonic_increasing:
        meteo_data = meteo_data.sort_index()
    return meteo_data


def load_sample_data():
    start = "20190801"
    end = "20191101"
    freq = "15min"
    index = pd.date_range(start, end, freq=freq, tz="CET")

    # Load satellite weather data: incident irradiation, direct irradiation, temperature, wind
    weather_data = pd.read_csv(ROOT / "data" / "meteo_sample.csv", index_col=0)
    weather_data["temp_air"] = weather_data["temp_air"] - 273.15
    weather_data.index = pd.to_datetime(weather_data.index)

    # Get ppm data
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

    return weather_data, pv_data


def get_pm_data(start_date="20210101",
                end_date="20210110",
                freq="H",
                site: str = "Grenoble Les Frenes",
                store_pkl: bool = True) -> pd.DataFrame:
    """

    Extract airborne particulate matter (PM) concentration with aerodynamic diameter less than 2.5 and 10 microns from
    data.gouv.

    Data available only from 2021 for Grenoble and is a granularity of an hour.

    :param start_date: date from which to start to extract particule matter concentrations
    :param end_date: date to which to stop to extract particule matter concentrations
    :param freq: frequency of the recipient (forward fill will be applied)
    :param site: which site to extract data for ?
    :param store_pkl: Store the data into pickle file ?

    :return: Dataframe with Datetime Index containing PM2.5 and PM10.
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
    config = import_config(ROOT / "data" / "config_site_0.json")
    pv_data = import_pv_data([ROOT / "data" / "pv_data_0.json"])
    meteo = import_weather_data([ROOT / "data" / "std_meteo_0.json"])

    import pprint

    pprint.pprint(config)
    print(pv_data.columns)
    print(meteo.columns)

    start = pd.to_datetime("20190801")
    end = pd.to_datetime("20191101")
    index = pd.date_range(start, end, freq="15min")
    pv_data.index = pd.to_datetime(pv_data.index)

    # Filter through dates and relevant columns
    pv_data_sample = pv_data.loc[start:end, ["Vdc@mpp-1-1", "Idc@mpp-1-1", "Pac1@inv-1"]]
    pv_data_sample = pv_data_sample.tz_localize('Etc/GMT-2').tz_convert("UTC")
    meteo_sample = meteo.loc[start:end, ["Gi%1-0-0", 'Gid%1-0-0', 'Gh', 'Ghc', 'Text', 'wind_speed', "rain_fall"]]
    meteo_sample = meteo_sample.tz_localize("UTC").iloc[:-1].tz_convert("UTC")

    # Renaming according to https://pvlib-python.readthedocs.io/en/stable/user_guide/variables_style_rules.html#variables-style-rules
    pv_data_sample = pv_data_sample.rename(columns={'Vdc@mpp-1-1': 'Vdc', 'Idc@mpp-1-1': 'Idc', 'Pac1@inv-1': 'Pac'})
    meteo_sample = meteo_sample.rename(columns={'Gi%1-0-0': 'poa_global', 'Gid%1-0-0': "poa_diffuse", "Gh": "ghi",
                                                "Ghc": "ghi_c", "Text": "temp_air"})

    meteo_sample.to_csv(ROOT / "data" / "meteo_sample.csv")
    pv_data_sample.to_csv(ROOT / "data" / "pv_data_sample.csv")

    meteo_sample = pd.read_csv(ROOT / "data" / "meteo_sample.csv", index_col=0)  # irradiation, temperature, wind
    pv_data_sample = pd.read_csv(ROOT / "data" / "pv_data_sample.csv", index_col=0)  # Vdc, Idc, Pac
