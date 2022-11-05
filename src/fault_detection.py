# Created by A. MATHIEU at 31/10/2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from pvlib.temperature import pvsyst_cell
from pvlib.solarposition import get_solarposition
from pvlib.pvsystem import pvwatts_dc

from pvanalytics.features.clipping import threshold

from src.simulate_underperformances import fixed_shading, clipping, bdiode_sc
from src.pv_models import fit_pvwatt, fit_vmp_king, vmp_king


def shading_detection(pdc: pd.Series,
                      pdc_estimated: pd.Series,
                      gi: pd.Series,
                      gid: pd.Series,
                      error_rel_thresh: float = 0,
                      error_window_thresh: float = 0.5,
                      gi_gid_ratio_thresh: float = 0.5,
                      window: str = "31D") -> pd.Series:
    """
    Flag shading according to the relative error from an estimated.
    Recurrent and ponctual relative errors are assessed.
    The algorithm has troubles to detect error on low irradiance.

    :param pdc: DC power [W/m2]
    :param pdc_estimated: Estimated DC power [W/m2]
    :param window: number of days to evaluate the recurrent pattern (center-window)

    :return: boolean pd.Series with flags when shading is detected
    """
    error = ((pdc_estimated - pdc) / pdc_s.abs()).clip(lower=-5, upper=5).to_frame("error_rel")
    error["time"] = error.index.time
    error["shading_flag"] = False
    error["gid_gi_ratio"] = gi / gid
    error["benchmark"] = np.nan

    for time in error["time"].unique():
        error_time = error[error["time"] == time].copy()
        if not error_time.empty:
            error.loc[error_time.index, "benchmark"] = \
                error_time["error_rel"].rolling(window, center=True).mean(min_count=1)

    error["shading_flag"] = (error["error_rel"] > error_rel_thresh) & \
                            (error["benchmark"] > error_window_thresh) & \
                            (error["error_rel"] > error["gid_gi_ratio"] * gi_gid_ratio_thresh)

    return error["shading_flag"]


def error_cluster(pdc: pd.Series,
                  pdc_estimated: pd.Series,
                  latitude: float = 48,
                  longitude: float = 3,
                  n_cluster=2) -> pd.DataFrame:
    """
    Return KNN-Cluster based on azimuth, elevation and error

    :param pdc: Real DC power timeserie (including shading)
    :param pdc_estimated: Estiamted DC power timeseries (without shading)
    :param latitude: Latitude of the site
    :param longitude: Longitude of the site
    :param n_cluster: Number of cluster to return

    :return: pd.Dataframe with cluster number in  "class" column
    """
    error = ((pdc_estimated - pdc) / pdc.abs()).clip(lower=-5, upper=5).to_frame("error_rel")
    solar_pos = get_solarposition(pdc.index, latitude, longitude, None)
    error["azimuth"] = solar_pos["azimuth"]
    error["elevation"] = solar_pos["elevation"]  # or apparent_elevation ?

    # normalization
    error_fit = error.dropna().copy()
    for col in error_fit.columns:
        error_fit[col] = (error_fit[col] - error_fit[col].mean()) / error_fit[col].std()
    mod = KMeans(n_clusters=n_cluster).fit(error_fit)
    error_class = pd.Series(mod.predict(error_fit), index=error_fit.index)

    error["class"] = error_class

    return error


def shut_module_detection(vdc: pd.Series,
                          vdc_estimated: pd.Series,
                          threshold: float = -20,
                          window: str = "21D"):
    diff = vdc_estimated.resample("D").mean() - vdc.resample("D").mean()
    diff.index = diff.index.date
    df = pd.Series(index=diff.index, dtype=float)
    for date in np.unique(diff.index):
        date_previous = date - pd.Timedelta(window)
        date_after = date + pd.Timedelta(window)
        df.loc[date] = diff.loc[(diff.index < date) & (diff.index >= date_previous)].mean(skipna=True) - \
                       diff.loc[(diff.index < date_after) & (diff.index >= date)].mean(skipna=True)

    flags_sc = (df < threshold)

    return flags_sc, df


if __name__ == '__main__':

    plot = True

    from src.data import load_sample_data
    from pvlib.pvsystem import retrieve_sam

    weather, pv = load_sample_data()
    idc, vdc, pdc, pac, = pv["Idc"], pv["Vdc"], pv["Pdc"], pv["Pac"]
    poa_global, poa_diffuse, temp_air = weather["poa_global"], weather["poa_diffuse"], weather["temp_air"]

    temp_cell = pvsyst_cell(poa_global, temp_air)
    index_fit = pv["Pdc"].dropna().index
    index_fit = index_fit[(index_fit < '2019-09-01 00:00:00+02:00') & (index_fit > '2019-08-01 00:00:00+02:00')]
    pdc_fit = pdc.reindex(index_fit)
    pdc0, gamma_pdc = fit_pvwatt(poa_global, pdc, temp_cell)
    pdc_estimated = pvwatts_dc(poa_global, temp_cell, pdc0, gamma_pdc)

    # Shading
    idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, latitude=44.109, longitude=0.272,
                                               temp_air=temp_air)
    shading_flags = shading_detection(pdc_s, pdc_estimated, poa_global, poa_diffuse, error_rel_thresh=0.1,
                                      error_window_thresh=0.5, gi_gid_ratio_thresh=0.3)
    pdc_s = pdc_s.fillna(pdc_estimated)
    error = error_cluster(pdc_s, pdc_estimated, latitude=48, longitude=3)

    if plot:
        pdc_estimated.plot()
        pdc_s.plot()
        pdc_s[shading_flags].plot(marker="+", linewidth=0)

        plt.figure()
        plt.scatter(error["azimuth"], error["elevation"], c=error["error_rel"], cmap="seismic")
        plt.figure()
        plt.scatter(error["azimuth"], error["elevation"], c=error["class"], cmap="seismic")

    ### Clipping
    pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
    idc_c, vdc_c, pdc_c, pac_c = clipping(gi, idc, vdc, pac, pac_max=42000, pv_params=pv_params, t_ext=t_ext)
    clipping_flags = threshold(pac_c)

    if plot:
        plt.figure()
        pac.plot()
        pac_c.plot()
        pac_c[clipping_flags].plot(marker="+", linewidth=0)

    ### Bypass diode
    sc_date = pd.Timestamp("20190901").tz_localize("CET")
    pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
    idc_sc, vdc_sc, pdc_sc, pac_sc = \
        bdiode_sc(poa_global, idc, vdc, pac, sc_date, pv_params, n_mod=21, ndiode=3, temp_air=temp_air)

    (c2, c3, beta, vmp_ref) = fit_vmp_king(poa_global, temp_cell=temp_cell, vmp=vdc)
    vdc_estimated = vmp_king(poa_global, temp_cell, c2, c3, beta, vmp_ref)
    threshold = - vdc.median() * 1 / 21 * 70 / 100
    flags_sc, df = shut_module_detection(vdc_sc, vdc_estimated, threshold)

    if plot:
        plt.figure()
        df.plot()

        plt.figure()
        vdc_sc.plot()
        flags_sc.index = pd.to_datetime(flags_sc.index).tz_localize("CET") + pd.Timedelta(hours=12)
        flags_sc = flags_sc.reindex(vdc_sc.index)
        flags_sc[flags_sc.isna()] = False
        vdc_estimated.plot()
        vdc_sc[flags_sc].reindex(vdc_sc.index).plot(marker="o", linewidth=0)
