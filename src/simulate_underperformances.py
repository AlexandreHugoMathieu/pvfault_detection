# Created by A. MATHIEU at 31/10/2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pvlib.solarposition import get_solarposition

from pvlib.soiling import hsu

from src.pv_models import fit_vmp_king, fit_imp_king, inv_eff_knn, imp_king, vmp_king, vi_curve_singlediode, get_Pmpp, \
    get_temp_cell


def scale_system_iv_curve(pv_params: dict, vdc: float, idc: float, poa_global: float, temp_cell: float) -> pd.DataFrame:
    """
    Scale  the IV curve of a module to the system according to Idc and Vdc

    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
                pvlib.pvsystem.retrieve_sam("CECmod")
    :param vdc: DC Voltage [V]
    :param idc: DC current [A]
    :param poa_global: Incident (effective) irradiation [W/m2]
    :param temp_cell: Cell temperature [°C]

    :return: pd.Dataframe with VI curve at the system level
    """
    vi_curve = vi_curve_singlediode(pv_params["alpha_sc"], pv_params["a_ref"], pv_params["I_L_ref"],
                                    pv_params["I_o_ref"], pv_params["R_sh_ref"], pv_params["R_s"],
                                    effective_irradiance=poa_global, temp_cell=temp_cell)
    _, impp_mod, vmpp_mod = get_Pmpp(vi_curve, VI_max=True)
    vi_curve["v_system"] = vi_curve["v"] * vdc / vmpp_mod
    vi_curve["i_system"] = vi_curve["i"] * idc / impp_mod

    return vi_curve


def king_ratios(poa_global0, temp_cell0, idc0, vdc0, poa_global1, temp_cell1):
    """Fit King models according to dataset0 and return dataset1/dataset0 model ratios"""

    (c1, alpha, imp_ref) = fit_imp_king(poa_global0, temp_cell0, idc0)
    (c2, c3, beta, vmp_ref) = fit_vmp_king(poa_global0, temp_cell0, vdc0)

    index = poa_global0.index
    imp_k0 = imp_king(poa_global0, temp_cell0, c1, alpha, imp_ref).reindex(index)
    imp_k1 = imp_king(poa_global1, temp_cell1, c1, alpha, imp_ref).reindex(index)
    vmp_k0 = vmp_king(poa_global0, temp_cell0, c2, c3, beta, vmp_ref).reindex(index)
    vmp_k1 = vmp_king(poa_global1, temp_cell1, c2, c3, beta, vmp_ref).reindex(index)

    imp_k_ratio = pd.Series(data=1, index=index)
    vmp_k_ratio = pd.Series(data=1, index=index)
    imp_k_ratio.loc[pd.notna(imp_k0)] = imp_k1 / imp_k0
    vmp_k_ratio.loc[pd.notna(vmp_k0)] = vmp_k1 / vmp_k0

    imp_k_ratio = imp_k_ratio.dropna().reindex(index)
    vmp_k_ratio = vmp_k_ratio.dropna().reindex(index)

    return imp_k_ratio, vmp_k_ratio


def shading_cond(azimuth: pd.Series, elevation: pd.Series, shade_azi_min=0, shade_azi_max=180, shade_alt=50,
                 noise_azimuth=(0, 0), noise_elevation=(0, 0)):
    azimuth_noise = pd.Series(np.random.normal(noise_azimuth[0], noise_azimuth[1], len(azimuth)), index=azimuth.index)
    elevation_noise = pd.Series(np.random.normal(noise_elevation[0], noise_elevation[1], len(elevation)),
                                index=elevation.index)
    shading_bool = (azimuth + azimuth_noise > shade_azi_min) & \
                   (azimuth + azimuth_noise < shade_azi_max) & \
                   (elevation + elevation_noise < shade_alt)
    return shading_bool


def fixed_shading(poa_global: pd.Series,
                  poa_diffuse: pd.Series,
                  idc: pd.Series,
                  vdc: pd.Series,
                  pac: pd.Series = None,
                  shade_azi_min: float = 0,
                  shade_azi_max: float = 180,
                  shade_alt: float = 50,
                  latitude: float = None,
                  longitude: float = None,
                  altitude: float = None,
                  azimuth: pd.Series = None,
                  elevation: pd.Series = None,
                  temp_air: pd.Series = None,
                  temp_cell: pd.Series = None):
    """
    Simulate fixed-horizon rectangular shading on operational current and voltage. Assume the shading is covering the
    whole installation (no partial shading)

    All Series should have the same Datetime Index.
    :param poa_global: Incident (effective) irradiation [W/m2]
    :param poa_diffuse: Incident diffuse irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param shade_azi_min: Shading azimuth minimum [°]
    :param shade_azi_max: Shading azimuth maximum [°]
    :param shade_alt: Shading altitude [°]
    :param latitude: Latitude of the installation to calculate sun's path azimuth and elevation (if azimuth and not
                    elevation directly provided)
    :param longitude: Longitude of the installation to calculate sun's path azimuth and elevation (if azimuth and not
                    elevation directly provided)
    :param altitude: (optional) Altitude of the installation to calculate sun's path azimuth and elevation (if not directly provided)
    :param azimuth: Azimuth to use for sun's path to evaluate if the shading is effective or not
    :param elevation: Elevation to use for sun's path to evaluate if the shading is effective or not
    :param temp_air: External temperature to use for estimating the cell temperature (if temp_cell not directly provided)
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading

    :return: DataFrames of Idc, Vdc, Pdc and Pac (if provided) with shading effect
    """
    # Extract  azimuth and elevation of the sun's path
    if ((latitude is None) or (longitude is None)) and ((azimuth is None) or (elevation is None)):
        raise ValueError("Please indicate either latitude/longitude or provide azimuth and elevation directly")
    if (azimuth is None) or (elevation is None):
        solar_pos = get_solarposition(idc.index, latitude, longitude, altitude)
        azimuth = solar_pos["azimuth"]
        elevation = solar_pos["elevation"]  # or apparent_elevation ?

    # Boolean condition to indicate if there is shading or not
    shading_bool = shading_cond(azimuth, elevation, shade_azi_min, shade_azi_max, shade_alt)

    # Roughly estimate the cell temperature if not directly provided
    temp_cell = get_temp_cell(temp_cell, temp_air, poa_global)

    # Fit King models and calculate the ratio between only diffuse and whole irradiation to later simulate shading
    imp_k_ratio, vmp_k_ratio = king_ratios(poa_global, temp_cell, idc, vdc, poa_diffuse, temp_cell)

    # Apply shading effect with the relative ratio from the modelled Imp and Vmp
    idc_shading = idc.copy()
    vdc_shading = vdc.copy()
    idc_shading.loc[shading_bool] = idc.loc[shading_bool] * imp_k_ratio
    vdc_shading.loc[shading_bool] = vdc.loc[shading_bool] * vmp_k_ratio
    pdc_shading = (idc_shading * vdc_shading).copy()

    if not pac is None:
        # Apply the new DC/AC ratio (which is function of the produced power)
        ratio = inv_eff_knn(pdc_fit=(idc * vdc), pac_fit=pac, pdc=pdc_shading)
        pac_shading = pac.copy()
        pac_shading.loc[shading_bool] = pdc_shading.loc[shading_bool] * ratio.loc[shading_bool]
    else:
        pac_shading = None

    return idc_shading, vdc_shading, pdc_shading, pac_shading


def clipping(poa_global: pd.Series,
             idc: pd.Series,
             vdc: pd.Series,
             pac: pd.Series,
             pac_max: float,
             pv_params: dict,
             temp_air: pd.Series = None,
             temp_cell: pd.Series = None):
    """
    Simulate inverter clipping according to an AC power limit on operational current and voltage.

    All Series should have the same Datetime Index.
    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param pac_max: maximum limit of the inverter
    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
                    pvlib.pvsystem.retrieve_sam("CECmod")
    :param temp_air: External temperature to use for estimating the cell temperature (if temp_cell not directly provided)
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading

    :return: DataFrames of Idc, Vdc, Pdc and Pac with clipping effect
    """
    # Prepare recipients
    idc_clipping = idc.copy()
    vdc_clipping = vdc.copy()
    pdc_clipping = idc_clipping * vdc_clipping
    pac_clipping = pac.copy()

    # Roughly estimate the cell temperature if not directly provided
    temp_cell = get_temp_cell(temp_cell, temp_air, poa_global)

    # Draw the IV curve for each point for Pac > pac_max and move the operating point to go under pac_max
    for idx in tqdm(pac_clipping[pac_clipping > pac_max].index, desc="Clipping"):
        vi_curve = scale_system_iv_curve(pv_params, vdc.loc[idx], idc.loc[idx], poa_global.loc[idx], temp_cell.loc[idx])

        # Assume the DC / AC efficiency is equal to the observed one, is constant over I/V and only affects the current
        inv_eff_pmpp = pac.loc[idx] / pdc_clipping.loc[idx]  # DC/AC efficiency
        vi_curve["i_system"] = vi_curve["i_system"] / inv_eff_pmpp  # DC side
        vi_curve["pdc"] = vi_curve["v_system"] * vi_curve["i_system"]
        vi_curve["pac"] = vi_curve["pdc"] * inv_eff_pmpp

        # Get the point with the maximum Pac under the pac_max limit and so vdc increases from the mpp point
        vi_max = vi_curve[(vi_curve["pac"] < pac_max) & (vi_curve["v_system"] > vdc.loc[idx])].sort_values("pac")
        vi_max = vi_max.iloc[-1]

        # Collect clipping data to recipients
        idc_clipping.loc[idx], vdc_clipping.loc[idx] = vi_max.loc["i_system"], vi_max.loc["v_system"]
        pdc_clipping.loc[idx], pac_clipping.loc[idx] = vi_max.loc["pdc"], pac_max

    return idc_clipping, vdc_clipping, pdc_clipping, pac_clipping


def soiling_effect(poa_global: pd.Series,
                   idc: pd.Series,
                   vdc: pd.Series,
                   pac: pd.Series,
                   rainfall: pd.Series,
                   pm2_5: pd.Series,
                   pm10: pd.Series,
                   tilt: float,
                   depo_veloc={'2_5': 0.0009, '10': 0.004},
                   cleaning_threshold=10,
                   rain_accum_period: pd.Timedelta = pd.Timedelta('24h'),
                   temp_air: pd.Series = None,
                   temp_cell: pd.Series = None):
    """
    Simulate soiling on operational current and voltage. It assumes soiling is uniforme and follows the hsu model from
    pvlib.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param rainfall: Rain accumulated in each time period. [mm]
    :param pm2_5: Airborne particulate matter (PM) concentration with aerodynamic diameter less than 2.5 microns. [g/m^3]
    :param pm10: Airborne particulate matter (PM) concentration with aerodynamic diameter less than 10 microns. [g/m^3]
    :param tilt: Tilt of the PV panels from horizontal. [degree]
    :param depo_veloc:  Deposition or settling velocity of particulates. [m/s]
    :param cleaning_threshold: Amount of rain in an accumulation period needed to clean the PV modules. [mm]
    :param rain_accum_period: Period for accumulating rainfall to check against `cleaning_threshold`
    :param temp_air: External temperature to use for estimating the cell temperature (if temp_cell not directly provided)
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading

    :return: DataFrames of Idc, Vdc, Pdc and Pac with soiling effect
    """

    # Compute soiling ratio according to the HSU model (pvlib) and assumes it uniformly applies on the irradiance
    s_ratio = hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10, depo_veloc, rain_accum_period)
    poa_global_soiling = poa_global * s_ratio

    # Roughly estimate the cell temperature if not directly provided
    temp_cell = get_temp_cell(temp_cell, temp_air, poa_global)

    # Fit King models and calculate the ratio between soiled and cleaned module
    imp_k_ratio, vmp_k_ratio = king_ratios(poa_global, temp_cell, idc, vdc, poa_global_soiling, temp_cell)

    # Apply soiling effect with the relative ratio from the modelled Imp and Vmp
    idc_soiling = idc.copy() * imp_k_ratio
    vdc_soiling = vdc.copy() * vmp_k_ratio
    pdc_soiling = (idc_soiling * vdc_soiling).copy()

    if not pac is None:
        # Apply the new DC/AC ratio (which is function of the produced power) to get AC power
        ratio = inv_eff_knn(pdc_fit=(idc * vdc), pac_fit=pac, pdc=pdc_soiling)
        pac_soiling = pdc_soiling * ratio
    else:
        pac_soiling = None

    return idc_soiling, vdc_soiling, pdc_soiling, pac_soiling


def bdiode_sc(poa_global: pd.Series,
              idc: pd.Series,
              vdc: pd.Series,
              pac: pd.Series,
              sc_date: pd.Timestamp,
              pv_params: dict,
              n_mod: float,
              ndiode: float = 1,
              n_cell_string: float = 3,
              temp_air: pd.Series = None,
              temp_cell: pd.Series = None):
    """
    Simulate the short circuit of some bypass diodes.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param sc_date: Date of the bypass short circuit
    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
                pvlib.pvsystem.retrieve_sam("CECmod")
    :param n_mod: Number of modules in the string
    :param ndiode: Number of short-circuited diode bypasses
    :param n_cell_string: Number of cell strings in a module
    :param temp_air: External temperature to use for estimating the cell temperature (if temp_cell not directly provided)
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading

    :return: DataFrames of Idc, Vdc, Pdc and Pac with bybass short circuit effect
    """

    # Prepare recipients
    idc_bypass_sc = idc.copy()
    vdc_bypass_sc = vdc.copy()
    pdc_bypass_sc = idc_bypass_sc * vdc_bypass_sc

    # Roughly estimate the cell temperature if not directly provided
    temp_cell = get_temp_cell(temp_cell, temp_air, poa_global)

    # Draw the IV curve for each point after the bypass shutdown, apply the short circuit and move the operating
    # point accordingly
    index = idc[(idc.index > sc_date) & (idc > 0)].dropna().index
    index = poa_global.reindex(index)[poa_global.reindex(index) > 0].index
    for idx in tqdm(index, desc="Bypass short-circuiting"):
        vi_curve = scale_system_iv_curve(pv_params, vdc.loc[idx], idc.loc[idx], poa_global.loc[idx], temp_cell.loc[idx])

        # Assume each module and each cell-string of each module has the same contribution + assume that the short
        # circuit of a bypass diode induces a drop in V proportional to the ratio of 1 cell-string to the total number
        # of cell-strings
        n_string = n_mod * n_cell_string
        vi_curve["v_system"] = vi_curve["v_system"] * (n_string - ndiode) / n_string
        pdc_bypass_sc.loc[idx], idc_bypass_sc.loc[idx], vdc_bypass_sc.loc[idx] = \
            get_Pmpp(vi_curve, VI_max=True, v_col="v_system", i_col="i_system")

    if not pac is None:
        # Apply the new DC/AC ratio (which is function of the produced power) to get AC power
        pac_bypass_sc = pac.copy()
        ratio = inv_eff_knn(pdc_fit=(idc * vdc), pac_fit=pac, pdc=pdc_bypass_sc)
        pac_bypass_sc.loc[pac_bypass_sc.index > sc_date] = pdc_bypass_sc.loc[pdc_bypass_sc.index > sc_date] * ratio
    else:
        pac_bypass_sc = None

    return idc_bypass_sc, vdc_bypass_sc, pdc_bypass_sc, pac_bypass_sc


def shading_azimuth_plot(azimuth: pd.Series = None, elevation: pd.Series = None, shade_azi_min: float = 0,
                         shade_azi_max: float = 180, shade_alt: float = 50):
    """ Plot shading as function of azimuth (x-axis) and elevation (y-axis) for a rectangular shade

    :param azimuth: Azimuth of the sun's path [°]
    :param elevation: Elevation of the sun's path [°]
    :param shade_azi_min: Shading azimuth minimum [°]
    :param shade_azi_max: Shading azimuth maximum [°]
    :param shade_alt: Shading altitude [°]

    """

    shading_bool = shading_cond(azimuth, elevation, shade_azi_min, shade_azi_max, shade_alt)

    fig = plt.figure()
    plt.plot(azimuth, elevation, ".", color="grey", label="Non-shaded")
    plt.plot(azimuth.loc[shading_bool], elevation.loc[shading_bool], ".", color="darkred", label="Shaded")
    plt.legend()

    return fig


def default_effect_plot(idc: pd.Series = None,
                        vdc: pd.Series = None,
                        pdc: pd.Series = None,
                        pac: pd.Series = None,
                        idc_default: pd.Series = None,
                        vdc_default: pd.Series = None,
                        pdc_default: pd.Series = None,
                        pac_default: pd.Series = None,
                        title: str = ""):
    idc_bool = (idc is not None) and (idc_default is not None)
    vdc_bool = (vdc is not None) and (vdc_default is not None)
    pdc_bool = (pdc is not None) and (pdc_default is not None)
    pac_bool = (pac is not None) and (pac_default is not None)

    fig, axes = plt.subplots(idc_bool + vdc_bool + pdc_bool + pac_bool, 1, sharex='all')
    fig.suptitle(title)
    i = 1
    if idc_bool:
        plt.subplot(idc_bool + vdc_bool + pdc_bool + pac_bool, 1, i)
        plt.title("Idc")
        plt.plot(idc, label="Non-defective")
        plt.plot(idc_default, label="Defective")
        i += 1
    if vdc_bool:
        plt.subplot(idc_bool + vdc_bool + pdc_bool + pac_bool, 1, i)
        plt.title("Vdc")
        plt.plot(vdc, label="Non-defective")
        plt.plot(vdc_default, label="Defective")
        i += 1
    if pdc_bool:
        plt.subplot(idc_bool + vdc_bool + pdc_bool + pac_bool, 1, i)
        plt.title("Pdc")
        plt.plot(pdc, label="Non-defective")
        plt.plot(pdc_default, label="Defective")
        i += 1
    if vdc_bool:
        plt.subplot(idc_bool + vdc_bool + pdc_bool + pac_bool, 1, i)
        plt.title("Pac")
        plt.plot(pac, label="Non-defective")
        plt.plot(pac_default, label="Defective")

    plt.legend()
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    plot = True
    store = True

    from src.data import load_sample_data
    from pvlib.pvsystem import retrieve_sam

    weather, pv = load_sample_data()
    idc, vdc, pdc, pac, = pv["Idc"], pv["Vdc"], pv["Pdc"], pv["Pac"]
    poa_global, poa_diffuse, temp_air = weather["poa_global"], weather["poa_diffuse"], weather["temp_air"]

    # Shading
    idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, latitude=44.109, longitude=0.272,
                                               temp_air=temp_air)

    # Clipping
    pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
    idc_c, vdc_c, pdc_c, pac_c = clipping(poa_global, idc, vdc, pac, pac_max=42000, pv_params=pv_params, temp_air=temp_air)

    # Soiling
    idc_soil, vdc_soil, pdc_soil, pac_soil = \
        soiling_effect(poa_global, idc, vdc, pac, weather["rain_fall"], weather["pm_2_5_g.m3"] * 20, weather["pm_10_g.m3"] * 20,
                       tilt=18, temp_air=temp_air)

    # Bypass short circuit
    pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
    sc_date = pd.Timestamp("20190901").tz_localize("CET")
    idc_sc, vdc_sc, pdc_sc, pac_sc = \
        bdiode_sc(poa_global, idc, vdc, pac, sc_date, pv_params, n_mod=21, ndiode=3, temp_air=temp_air)

    if plot:
        _ = default_effect_plot(idc, vdc, pdc, pac, idc_s, vdc_s, pdc_s, pac_s, "Shading")
        _ = default_effect_plot(idc, vdc, pdc, pac, idc_c, vdc_c, pdc_c, pac_c, "Clipping")
        _ = default_effect_plot(idc, vdc, pdc, pac, idc_soil, vdc_soil, pdc_soil, pac_soil, "Soiling")
        _ = default_effect_plot(idc, vdc, pdc, pac, idc_sc, vdc_sc, pdc_sc, pac_sc, "Short-circuit")

    if store:
        from src.config import ROOT
        pv_shading = pv.copy()
        pv_shading["Idc"], pv_shading["Vdc"], pv_shading["Pdc"], pv_shading["Pac"] = idc_s, vdc_s, pdc_s, pac_s
        pv_shading.to_csv(ROOT / "data" / "pv_data_shading.csv")

        pv_clipping = pv.copy()
        pv_clipping["Idc"], pv_clipping["Vdc"], pv_clipping["Pdc"], pv_clipping["Pac"] = idc_c, vdc_c, pdc_c, pac_c
        pv_clipping.to_csv(ROOT / "data" / "pv_data_clipping.csv")

        pv_shortcircuit = pv.copy()
        pv_shortcircuit["Idc"], pv_shortcircuit["Vdc"], pv_shortcircuit["Pdc"], pv_shortcircuit["Pac"] = \
            idc_sc, vdc_sc, pdc_sc, pac_sc
        pv_shortcircuit.to_csv(ROOT / "data" / "pv_data_shortcircuit.csv")

        ### All defaults combined
        # shading
        idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, latitude=44.109,
                                                   longitude=0.272,
                                                   temp_air=temp_air)
        # clipping
        pv_params = retrieve_sam('cecmod')['Instalaciones_Pevafersa_IP_VAP230']  # panel in the field
        idc_c, vdc_c, pdc_c, pac_c = clipping(poa_global, idc_s, vdc_s, pac_s, pac_max=42000, pv_params=pv_params,
                                              temp_air=temp_air)
        # short circuit
        sc_date = pd.Timestamp("20190901").tz_localize("CET")
        idc_all, vdc_all, pdc_all, pac_all = \
            bdiode_sc(poa_global, idc_c, vdc_c, pac_c, sc_date, pv_params, n_mod=21, ndiode=3, temp_air=temp_air)

        pv_data_all_defaults = pv.copy()
        pv_data_all_defaults["Idc"], pv_data_all_defaults["Vdc"], pv_data_all_defaults["Pdc"], pv_data_all_defaults[
            "Pac"] = idc_sc, vdc_sc, pdc_sc, pac_sc
        pv_data_all_defaults.to_csv(ROOT / "data" / "pv_data_all_defaults.csv")
