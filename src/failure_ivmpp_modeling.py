# Created by A. MATHIEU at 31/10/2022
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pvlib.soiling import hsu

from src.pv_models import fit_imp_vmp_king, inv_eff_knn, imp_king, vmp_king, iv_curve_singlediode, get_Pmpp


def scale_system_iv_curve(pv_params: dict,
                          vdc: float,
                          idc: float,
                          g_poa_effective: float,
                          temp_cell: float) -> pd.DataFrame:
    """
    Scale the module IV curve to the whole system via Idc and Vdc

    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
            pvlib.pvsystem.retrieve_sam("CECmod")
    :param vdc: DC Voltage [V]
    :param idc: DC current [A]
    :param g_poa_effective: In plane (effective) irradiation [W/m2]
    :param temp_cell: Cell temperature [°C]

    :return: IV curve: current [A] and voltage [V]
    """

    vi_curve = iv_curve_singlediode(pv_params["alpha_sc"], pv_params["a_ref"], pv_params["I_L_ref"],
                                    pv_params["I_o_ref"], pv_params["R_sh_ref"], pv_params["R_s"],
                                    effective_irradiance=g_poa_effective, temp_cell=temp_cell)

    # Scale the curve by scaling impp and vmpp at vdc and idc (Neglect all extra system possible losses)
    _, impp_mod, vmpp_mod = get_Pmpp(vi_curve, VI_max=True)
    vi_curve["v_system"] = vi_curve["v"] * vdc / vmpp_mod
    vi_curve["i_system"] = vi_curve["i"] * idc / impp_mod

    return vi_curve


def shading_cond(azimuth: pd.Series,
                 elevation: pd.Series,
                 shade_azi_min: float = 0,
                 shade_azi_max: float = 180,
                 shade_alt: float = 50):
    """
    Return a boolean pd.Series to check if the installation is under the shadow of a fixed-rectangular shape or not

    :param azimuth: Azimuth of the sun position pd.Series
    :param elevation: Sky elevation of the sun position pd.Series
    :param shade_azi_min: Minimum shade azimuth of the rectangular shape [°]
    :param shade_azi_max: Maximum shade azimuth of the rectangular shape [°]
    :param shade_alt: Maximum shade elevation of the rectangular shape [°]

    :return: boolean series to indicate the presence of shadow
    """
    shading_bool = (azimuth > shade_azi_min) & (azimuth < shade_azi_max) & (elevation < shade_alt)
    return shading_bool


def fixed_shading(poa_global: pd.Series,
                  poa_diffuse: pd.Series,
                  idc: pd.Series,
                  vdc: pd.Series,
                  pac: pd.Series = None,
                  azimuth: pd.Series = None,
                  elevation: pd.Series = None,
                  temp_cell: pd.Series = None,
                  shade_azi_min: float = 0,
                  shade_azi_max: float = 180,
                  shade_alt: float = 50):
    """
    Simulate the effects of a fixed-horizon rectangular shading on operational current and voltage.

    Assume the shading is covering the whole installation (no partial shading)

    All Series should have the same Datetime Index.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param poa_diffuse: Incident diffuse irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param azimuth: Azimuth to use for sun's path to evaluate if the shading is effective or not
    :param elevation: Elevation to use for sun's path to evaluate if the shading is effective or not
    :param temp_cell: Cell temperature to model imp and vmp variations under shading
    :param shade_azi_min: Minimum shade azimuth of the rectangular shape [°]
    :param shade_azi_max: Maximum shade azimuth of the rectangular shape [°]
    :param shade_alt: Maximum shade elevation of the rectangular shape [°]

    :return: Operational variables including shading effect
    """
    # Boolean condition to indicate if there is shading or not
    shading_bool = shading_cond(azimuth, elevation, shade_azi_min, shade_azi_max, shade_alt)

    # Fit King models and calculate the ratio between only diffuse and whole irradiation to later simulate shading
    (c1, alpha, i_ref, c2, c3, beta, v_ref) = fit_imp_vmp_king(poa_global, temp_cell, idc, vdc)
    imp_ratio = imp_king(poa_diffuse, temp_cell, c1, alpha, i_ref) / \
                imp_king(poa_global, temp_cell, c1, alpha, i_ref)
    vmp_ratio = vmp_king(poa_diffuse, temp_cell, c2, c3, beta, v_ref) / \
                vmp_king(poa_global, temp_cell, c2, c3, beta, v_ref)

    # Apply shading effect with the relative ratio from the modelled Imp and Vmp
    idc_shading = idc.copy()
    vdc_shading = vdc.copy()
    idc_shading.loc[shading_bool] = idc.loc[shading_bool] * imp_ratio.fillna(1)
    vdc_shading.loc[shading_bool] = vdc.loc[shading_bool] * vmp_ratio.fillna(1)
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
             temp_cell: pd.Series,
             pac_max: float,
             pv_params: dict):
    """
    Simulate inverter clipping according to an AC power limit on operational current and voltage.

    All Series should have the same Datetime Index.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param temp_cell: Cell temperature to model imp and vmp variations under clipping
    :param pac_max: maximum limit of the inverter
    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
                    pvlib.pvsystem.retrieve_sam("CECmod")

    :return: Operational variables including clipping effect
    """
    # Prepare recipients
    idc_clipping = idc.copy()
    vdc_clipping = vdc.copy()
    pdc_clipping = idc_clipping * vdc_clipping
    pac_clipping = pac.copy()

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
                   temp_cell: pd.Series,
                   tilt: float,
                   depo_veloc={'2_5': 0.0009, '10': 0.004},
                   cleaning_threshold=10,
                   rain_accum_period: pd.Timedelta = pd.Timedelta('24h')):
    """
    Simulate soiling on operational current and voltage.

    It assumes soiling is uniform over the system and follows the hsu model from pvlib.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param rainfall: Rain accumulated in each time period. [mm]
    :param pm2_5: Airborne particulate matter (PM) concentration with aerodynamic diameter less than 2.5 microns. [g/m^3]
    :param pm10: Airborne particulate matter (PM) concentration with aerodynamic diameter less than 10 microns. [g/m^3]
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading.
    :param tilt: Tilt of the PV panels from horizontal. [degree]
    :param depo_veloc:  Deposition or settling velocity of particulates. [m/s]
    :param cleaning_threshold: Amount of rain in an accumulation period needed to clean the PV modules. [mm]
    :param rain_accum_period: Period for accumulating rainfall to check against cleaning_threshold

    :return: Operational variables including soiling effect
    """

    # Compute soiling ratio according to the HSU model (pvlib) and assumes it uniformly applies on the irradiance
    s_ratio = hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10, depo_veloc, rain_accum_period)
    poa_global_soiling = poa_global * s_ratio

    # Fit King models and calculate the ratio between soiled and cleaned module
    (c1, alpha, i_ref, c2, c3, beta, v_ref) = fit_imp_vmp_king(poa_global, temp_cell, idc, vdc)
    imp_ratio = imp_king(poa_global_soiling, temp_cell, c1, alpha, i_ref) / \
                imp_king(poa_global, temp_cell, c1, alpha, i_ref)
    vmp_ratio = vmp_king(poa_global_soiling, temp_cell, c2, c3, beta, v_ref) / \
                vmp_king(poa_global, temp_cell, c2, c3, beta, v_ref)

    # Apply soiling effect with the relative ratio from the modelled Imp and Vmp
    idc_soiling = idc.copy() * imp_ratio.fillna(1)
    vdc_soiling = vdc.copy() * vmp_ratio.fillna(1)
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
              temp_cell: pd.Series,
              sc_date: pd.Timestamp,
              pv_params: dict,
              n_mod: float,
              ndiode: float = 1,
              n_cell_string: float = 3):
    """
    Simulate the short circuit of some bypass diodes.

    :param poa_global: Incident (effective) irradiation [W/m2]
    :param idc: DC current [A]
    :param vdc: DC Voltage [V]
    :param pac: AC power [W]
    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading
    :param sc_date: Date of the bypass short circuit
    :param pv_params: dictionary with module characteristics, can be extracted with pvlib thanks to
                pvlib.pvsystem.retrieve_sam("CECmod")
    :param n_mod: Number of modules in the string
    :param ndiode: Number of short-circuited diode bypasses
    :param n_cell_string: Number of cell strings in a module

    :return: DataFrames of Idc, Vdc, Pdc and Pac with bybass short circuit effect
    """

    # Prepare recipients
    idc_bypass_sc = idc.copy()
    vdc_bypass_sc = vdc.copy()
    pdc_bypass_sc = idc_bypass_sc * vdc_bypass_sc

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
    """Plot the operational variables with/without default"""
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

    from src.config import ROOT
    from src.data import load_sample_data
    from pvlib.pvsystem import retrieve_sam

    weather, pv, azimuth, elevation, temp_cell, secret = load_sample_data()
    idc, vdc, pdc, pac, = pv["Idc"], pv["Vdc"], pv["Pdc"], pv["Pac"]
    poa_global, poa_diffuse = weather["poa_global"], weather["poa_diffuse"]

    # Shading
    idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, azimuth, elevation, temp_cell)

    # Clipping
    pv_params = retrieve_sam('cecmod')[secret.loc["panel_name"]]  # panel in the field
    idc_c, vdc_c, pdc_c, pac_c = clipping(poa_global, idc, vdc, pac, temp_cell, pac_max=42000, pv_params=pv_params)

    # Soiling
    idc_soil, vdc_soil, pdc_soil, pac_soil = \
        soiling_effect(poa_global, idc, vdc, pac, weather["rain_fall"], weather["pm_2_5_g.m3"] * 20,
                       weather["pm_10_g.m3"] * 20, temp_cell, secret.loc["tilt"])

    # Bypass short circuit
    pv_params = retrieve_sam('cecmod')[secret.loc["panel_name"]]  # panel in the field
    sc_date = pd.Timestamp("20190901").tz_localize("CET")
    idc_sc, vdc_sc, pdc_sc, pac_sc = \
        bdiode_sc(poa_global, idc, vdc, pac, temp_cell, sc_date, pv_params, n_mod=secret.loc["n_module"],
                  ndiode=secret.loc["n_diode"])

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
        idc_s, vdc_s, pdc_s, pac_s = fixed_shading(poa_global, poa_diffuse, idc, vdc, pac, azimuth, elevation,
                                                   temp_cell)
        # clipping
        pv_params = retrieve_sam('cecmod')[secret.loc["panel_name"]]  # panel in the field
        idc_c, vdc_c, pdc_c, pac_c = clipping(poa_global, idc_s, vdc_s, pac_s, temp_cell, pac_max=42000,
                                              pv_params=pv_params)
        # short circuit
        sc_date = pd.Timestamp("20190901").tz_localize("CET")
        idc_all, vdc_all, pdc_all, pac_all = bdiode_sc(poa_global, idc_c, vdc_c, pac_c, temp_cell, sc_date, pv_params,
                                                       n_mod=secret.loc["n_module"], ndiode=secret.loc["n_diode"])

        pv_data_all = pv.copy()
        pv_data_all["Idc"], pv_data_all["Vdc"], pv_data_all["Pdc"], pv_data_all["Pac"] = idc_sc, vdc_sc, pdc_sc, pac_sc
        pv_data_all.to_csv(ROOT / "data" / "pv_data_all_defaults.csv")
