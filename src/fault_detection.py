"""This scripts collects simple tools to detect failures"""
# Created by A. MATHIEU at 31/10/2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from pvlib.pvsystem import pvwatts_dc

from pvanalytics.features.clipping import threshold

from src.failure_ivmpp_modeling import fixed_shading, clipping, bdiode_sc
from src.pv_models import fit_pvwatt, fit_vmp_king, vmp_king


def shading_detection(pdc: pd.Series,
                      pdc_estimated: pd.Series,
                      poa_global: pd.Series,
                      poa_diffuse: pd.Series,
                      error_rel_thresh: float = 0,
                      error_window_thresh: float = 0.5,
                      gib_gid_ratio_thresh: float = 0.5,
                      window: str = "31D") -> pd.Series:
    """
    Flag shading based on DC power and beam/global irradiation.

    The algorithm assesses  punctual, recurrent relative errors as well as diffuse & total irradiation against thresholds.
    If the three conditions are met, it flags the datetime as shading.

    Note that the algorithm has troubles to detect error on low irradiance.

    :param pdc: DC power [W/m2]
    :param pdc_estimated: Estimated DC power [W/m2]
    :param window: number of days to evaluate the recurrent pattern on the relative errr (centered-window)
    :param poa_global: Incident (effective) irradiation [W/m2]
    :param poa_diffuse: Incident diffuse irradiation [W/m2]
    :param error_rel_thresh: punctual relative error threshold
    :param error_window_thresh: recurrent relative error threshold
    :param gib_gid_ratio_thresh: gib/gid threshold to evaluate the error against
    :param window: number of days to take into account to evaluate the recurrent error

    :return: boolean pd.Series with flags when shading is detected
    """

    # Prepare recipient
    error = ((pdc_estimated - pdc) / pdc_s.abs()).clip(lower=-5, upper=5).to_frame("error_rel")
    error["gib_gid_ratio"] = (poa_global - poa_diffuse) / poa_diffuse
    error["recurrent_error"] = np.nan
    error["shading_flag"] = False

    # For each "time", calculate the error on a centered-window based on a number of days to evaluate the recurrent pattern
    error["time"] = error.index.time
    error["recurrent_error"] = error[["error_rel", "time"]].groupby("time").transform(
        lambda x: x.rolling(window, center=True).mean(min_count=1))

    # If punctual error, recurrent error and gi/gid ratio conditions are met, flag it as shading
    error["shading_flag"] = (error["error_rel"] > error_rel_thresh) & \
                            (error["recurrent_error"] > error_window_thresh) & \
                            (error["error_rel"] > error["gib_gi_ratio"] * gib_gid_ratio_thresh)

    return error["shading_flag"]


def error_cluster(pdc: pd.Series,
                  pdc_estimated: pd.Series,
                  features: pd.DataFrame,
                  n_cluster=2) -> pd.DataFrame:
    """
    Return KNN-Cluster based on features and relative error

    :param pdc: Real DC power timeserie (including shading for example)
    :param pdc_estimated: Estimated DC power timeseries (without shading for example)
    :param features: Features of the dataset to train the KNN model on
    :param n_cluster: Number of cluster to return

    :return: pd.Dataframe with cluster number in  "class" column
    """
    error = ((pdc_estimated - pdc) / pdc.abs()).clip(lower=-5, upper=5).to_frame("error_rel")
    for col in features.columns:
        error[col] = features[col]

    # Feature normalization
    error_fit = error.dropna().copy()
    for col in error_fit.columns:
        error_fit[col] = (error_fit[col] - error_fit[col].mean()) / error_fit[col].std()

    # Fit the model on the whole provided training dataset and predict on the same set
    mod = KMeans(n_clusters=n_cluster).fit(error_fit)
    error["class"] = pd.Series(mod.predict(error_fit), index=error_fit.index)

    return error


def short_circuit_detection(vdc: pd.Series,
                            vdc_estimated: pd.Series,
                            threshold: float = 20,
                            window: str = "15D"):
    """
    Detect a mean change in the error indicator.

    For each date, the error indicator is equal to the difference of the means over a time-window before and after
    the date of the error between the daily-averaged vdc_estimated and vdc.

    :param vdc: DC Voltage [V]
    :param vdc_estimated: Estimated DC voltage [V]
    :param threshold: All timestamps for which the indicator goes over the threshold are flagged as short_circuit
                        detection
    :param window: number of days taken to average the daily error before and after each date

    :return: Flag when the Vdc-error indicator is over the threshold + the indicator itself in a pd.Series
    """
    diff = vdc_estimated.resample("D").mean() - vdc.resample("D").mean()

    diff.index = diff.index.date
    error_kpi = pd.Series(index=diff.index, dtype=float)
    for date in np.unique(diff.index):
        date_previous = date - pd.Timedelta(window)
        date_after = date + pd.Timedelta(window)
        error_kpi.loc[date] = diff.loc[(diff.index < date_after) & (diff.index >= date)].mean(skipna=True) - \
                              diff.loc[(diff.index < date) & (diff.index >= date_previous)].mean(skipna=True)

    flags_sc = (error_kpi > threshold)

    return flags_sc, error_kpi


if __name__ == '__main__':

    plot = True

    from src.data import load_sample_data
    from pvlib.pvsystem import retrieve_sam

    weather, pv, azimuth, elevation, temp_cell, secret = load_sample_data()
    idc, vdc, pdc, pac, = pv["Idc"], pv["Vdc"], pv["Pdc"], pv["Pac"]
    poa_global, poa_diffuse, temp_air = weather["poa_global"], weather["poa_diffuse"], weather["temp_air"]

    # Build an Pdc estimate
    index_fit = pv["Pdc"].dropna().index
    index_fit = index_fit[(index_fit < '2019-09-01 00:00:00+02:00') & (index_fit > '2019-08-01 00:00:00+02:00')]
    pdc_fit = pdc.reindex(index_fit)
    pdc0, gamma_pdc = fit_pvwatt(poa_global, pdc, temp_cell)
    pdc_estimated = pvwatts_dc(poa_global, temp_cell, pdc0, gamma_pdc)

    # Shading
    idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, azimuth, elevation, temp_cell)
    shading_flags = shading_detection(pdc_s, pdc_estimated, poa_global, poa_diffuse, error_rel_thresh=0.1,
                                      error_window_thresh=0.5, gi_gid_ratio_thresh=0.5)
    pdc_s = pdc_s.fillna(pdc_estimated)
    features = pd.concat([azimuth.to_frame("azimuth"), elevation.to_frame("elevation")], axis=1).dropna()
    error = error_cluster(pdc_s, pdc_estimated, features)

    if plot:
        pdc_estimated.plot()
        pdc_s.plot()
        pdc_s[shading_flags].plot(marker="+", linewidth=0)

        plt.figure()
        plt.scatter(error["azimuth"], error["elevation"], c=error["class"], cmap="seismic")

    ### Clipping
    pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
    idc_c, vdc_c, pdc_c, pac_c = clipping(poa_global, idc, vdc, pac, temp_cell, pac_max=42000, pv_params=pv_params)
    clipping_flags = threshold(pac_c)

    if plot:
        plt.figure()
        pac.plot()
        pac_c.plot()
        pac_c[clipping_flags].plot(marker="+", linewidth=0)

    ### Bypass diode
    sc_date = pd.Timestamp("20190901").tz_localize("CET")
    pv_params = retrieve_sam('cecmod')[secret.loc["panel_name"]]  # panel in the field
    idc_sc, vdc_sc, pdc_sc, pac_sc = \
        bdiode_sc(poa_global, idc, vdc, pac, temp_cell, sc_date, pv_params, n_mod=secret.loc["n_module"],
                  ndiode=secret.loc["n_diode"])

    (c2, c3, beta, vmp_ref) = fit_vmp_king(poa_global, temp_cell=temp_cell, vmp=vdc)
    vdc_estimated = vmp_king(poa_global, temp_cell, c2, c3, beta, vmp_ref)

    threshold = vdc.median() * 1 / secret.loc["n_module"] * 70 / 100
    flags_sc, df = short_circuit_detection(vdc_sc, vdc_estimated, threshold)

    if plot:
        plt.figure()
        vdc_sc.plot()
        flags_sc.index = pd.to_datetime(flags_sc.index).tz_localize("CET") + pd.Timedelta(hours=12)
        flags_sc = flags_sc.reindex(vdc_sc.index)
        flags_sc[flags_sc.isna()] = False
        vdc_estimated.plot()
        vdc_sc[flags_sc].reindex(vdc_sc.index).plot(marker="o", linewidth=0)
