# Created by A. MATHIEU at 30/10/2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
from pvlib.pvsystem import calcparams_desoto, singlediode, pvwatts_dc

from src.data_utils import get_temp_cell


def inv_eff_knn(pdc_fit: pd.Series, pac_fit: pd.Series, pdc: pd.Series, n_neighbors: int = 100) -> pd.Series:
    """
    Fit a KNN model and predict the AC/DC efficiency

    Parameters
    ----------
    pdc_fit: DC power [W] to fit the model
    pac_fit: AC power [W] to fit the model
    pdc: DC power [W] to predict the ratio for
    n_neighbors: Number of neighbors in the KNN model

    Returns
    -------
    Predicted ratios from the KNN model
    """
    # Prepare the index with no nans and 0s to fit the model
    index_fit = pdc_fit.dropna().index.intersection(pac_fit.dropna().index)
    index_fit = pdc_fit.reindex(index_fit)[(pdc_fit.reindex(index_fit) > 0) & (pac_fit.reindex(index_fit) > 0)].index
    pac_fit = pac_fit.reindex(index_fit)
    pdc_fit = pdc_fit.reindex(index_fit)

    # Fit model
    knn_mod = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_mod = knn_mod.fit(pdc_fit.to_frame(""), (pac_fit / pdc_fit).values)

    # Predict ratios
    ratio = pd.Series(data=knn_mod.predict(pdc.dropna().to_frame("")), index=pdc.dropna().index)
    ratio = ratio.reindex(pdc.index)

    return ratio


def imp_teh(goa_effective: pd.Series, temp_cell: pd.Series,
            alpha: float, imp_ref: float,
            temp_stc: float = 25):
    """
    Estimate Imp according to Christian Jun Qian Teh's article.

     References
    ----------

     .. [1] C. J. Q. Teh, M. Drieberg, S. Soeung and R. Ahmad, "Simple PV Modeling Under Variable Operating Conditions,"
     in IEEE Access, vol. 9, pp. 96546-96558, 2021, doi: 10.1109/ACCESS.2021.3094801.
    """
    imp = imp_ref * (goa_effective / 1000) * (1 + alpha * (temp_cell - temp_stc))

    return imp


def imp_king(goa_effective: pd.Series,
             temp_cell: pd.Series,
             c1: float,
             alpha: float,
             imp_ref: float,
             reference_temperature: float = 25,
             reference_irradiance: float = 1000) -> pd.Series:
    """

    Parameters
    ----------
    goa_effective: Irradiance reaching the module's cells, after reflections and  adjustment for spectrum. [W/m2]
    temp_cell:  Cell temperature [C].
    c1: empirically determined coefficient
    alpha: Maximum power current temperature coefficient at reference condition (1/C)
    imp_ref: Power current reference [A]
    reference_temperature:  Reference temperature at STC conditions [C]
    reference_irradiance :  Reference Irradiance at STC conditions [W/m2]

    Returns
    -------
    imp: pd.Series
        Current at maximum power point

    References
    -------
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    """
    Ee = goa_effective / reference_irradiance
    imp = imp_ref * (Ee + c1 * (Ee ** 2)) * (1 + alpha * (temp_cell - reference_temperature))

    return imp


def vmp_king(goa_effective: pd.Series,
             temp_cell: pd.Series,
             c2: float,
             c3: float,
             beta: float,
             vmp_ref: float,
             reference_temperature: float = 25,
             reference_irradiance: float = 1000):
    """
    Estimate Vmp at maximum power according to King's model.
    
    Parameters
    ----------
    goa_effective: Irradiance reaching the module's cells, after reflections and  adjustment for spectrum. [W/m2]
    temp_cell:  Cell temperature [C].
    c2: empirically determined coefficient
    c3: empirically determined coefficient
    beta: Open circuit voltage temperature coefficient at reference condition (V/C)
    vmp_ref: Power voltage reference [V]
    reference_temperature  :  Reference temperature at STC conditions [C]
    reference_irradiance  :  Reference Irradiance at STC conditions [W/m2]

    Returns
    -------
    vmp: pd.Series
        Voltage at maximum power point

    References
    -------
    .. [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
       Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
       NM.

    """
    Ee = goa_effective / reference_irradiance

    temp_cell_K = temp_cell + 273.15
    vmp = vmp_ref + \
          c2 * temp_cell_K * np.log(Ee) + \
          c3 * (temp_cell_K * np.log(Ee)) ** 2 + \
          beta * goa_effective * (temp_cell - reference_temperature)

    return vmp


def fit_imp_king(g_poa_effective: pd.Series,
                 temp_cell: pd.Series,
                 imp: pd.Series):
    """
    Empirically fit Imp King model's parameters with brute force method relying on scipy.optimize.curve_fit

    Parameters
    ----------
    goa_effective: Irradiance reaching the module's cells, after reflections and  adjustment for spectrum. [W/m2]
    temp_cell:  Cell temperature [C].
    imp: Current at the maximum power point [A]

    Returns
    -------
    King imp's parameter inputs: c1, alpha, imp_ref
    """

    # Make sure there is no Nans
    index_fit = imp.dropna().index.intersection(g_poa_effective.dropna().index).intersection(temp_cell.dropna().index)
    imp_fit = imp.reindex(index_fit)
    g_poa_effective_fit = g_poa_effective.reindex(index_fit)
    temp_cell_fit = temp_cell.reindex(index_fit)

    # Fit King's imp model
    def imp_to_fit(X, c1, alpha, imp_ref):
        g_poa_effective, temp_cell = X
        return imp_king(g_poa_effective, temp_cell, c1, alpha, imp_ref)

    c1, alpha, imp_ref = curve_fit(imp_to_fit, (g_poa_effective_fit, temp_cell_fit), (imp_fit))[0]

    return (c1, alpha, imp_ref)


def fit_vmp_king(g_poa_effective: pd.Series, temp_cell: pd.Series, vmp: pd.Series):
    """
    Empirically fit Vmp King model's parameters with brute force method relying on scipy.optimize.curve_fit

    Parameters
    ----------
    goa_effective: Irradiance reaching the module's cells, after reflections and  adjustment for spectrum. [W/m2]
    temp_cell:  Cell temperature [C].
    imp: Voltage at the maximum power point [A]

    Returns
    -------
    King vmp's parameter inputs: c2, c3, beta, vmp_ref
    """

    # Make sure there is no Nans and no irradiation with 0s
    index_fit = vmp.dropna().index.intersection(g_poa_effective.dropna().index).intersection(temp_cell.dropna().index)
    index_fit = vmp.reindex(index_fit)[vmp.reindex(index_fit) != 0].index
    index_fit = g_poa_effective.reindex(index_fit)[g_poa_effective.reindex(index_fit) != 0].index
    vmp_fit = vmp.reindex(index_fit)
    g_poa_effective_fit = g_poa_effective.reindex(index_fit)
    temp_cell_fit = temp_cell.reindex(index_fit)

    # Fit vmp model
    def vmp_to_fit(X, c2, c3, beta, vmp_ref):
        g_poa_effective, temp_cell = X
        return vmp_king(g_poa_effective, temp_cell, c2, c3, beta, vmp_ref)

    c2, c3, beta, vmp_ref = curve_fit(vmp_to_fit, (g_poa_effective_fit, temp_cell_fit), (vmp_fit))[0]

    return (c2, c3, beta, vmp_ref)


def fit_pvwatt(g_poa_effective: pd.Series,
               pdc: pd.Series,
               temp_cell: pd.Series = None,
               temp_air: pd.Series = None):
    """
    Empirically pvwatt model's parameters with brute force method relying on scipy.optimize.curve_fit

    Parameters
    ----------
    goa_effective: Irradiance reaching the module's cells, after reflections and  adjustment for spectrum. [W/m2]
    imp: Voltage at the maximum power point [A]
    pdc: DC power at maximum power point [W]
    temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading [C]
    temp_air: External temperature to use for estimating the cell temperature (if temp_cell not directly provided) [C]

    Returns
    -------
    Pvwatt Pdc model's parameter inputs: pdc0, gamma_pdc

    """
    # Get temperature for PV watt model
    temp_cell = get_temp_cell(temp_cell, temp_air, g_poa_effective)

    # Make sure there is no Nans and no irradiation with 0s
    index_fit = pdc.dropna().index.intersection(g_poa_effective.dropna().index).intersection(temp_cell.dropna().index)
    index_fit = g_poa_effective.reindex(index_fit)[g_poa_effective.reindex(index_fit) != 0].index
    pdc_fit = pdc.reindex(index_fit)
    g_poa_effective_fit = g_poa_effective.reindex(index_fit)
    temp_cell_fit = temp_cell.reindex(index_fit)

    def pvwatt_to_fit(X, pdc0, gamma_pdc):
        g_poa_effective, temp_cell = X
        pdc = pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.0)
        return pdc

    pdc0, gamma_pdc = curve_fit(pvwatt_to_fit, (g_poa_effective_fit, temp_cell_fit), pdc_fit, p0)[0]

    return pdc0, gamma_pdc


def vi_curve_singlediode(alpha_sc: float, a_ref: float, I_L_ref: float, I_o_ref: float, R_sh_ref: float, R_s: float,
                         n_points: float = 1000, effective_irradiance: float = 1000, temp_cell: float = 25,
                         EgRef: float = 1.121,
                         dEgdT: float = -0.0002677) -> pd.DataFrame:
    """
    Draw the IV curve according to DeSoto method and the single Diode model.

    Parameters
    ----------
    alpha_sc: The short-circuit current temperature coefficient of the  module in units of A/C.
    a_ref: The product of the usual diode ideality factor (n, unitless),
         number of cells in series (Ns), and cell thermal voltage at reference
         conditions, in units of V.
    IL: The light-generated current (or photocurrent) at reference conditions, in amperes.
    I_o_ref: The dark or diode reverse saturation current at reference conditions, in amperes.
    Rs: The series resistance at reference conditions, in ohms.
    Rsh: The shunt resistance at reference conditions, in ohms.
    n_points: Number of points in the desired IV curve
    effective_irradiance: The irradiance (W/m2) that is converted to photocurrent.
    temp_cell:  The average cell temperature of cells within a module in C.
    EgRef: The energy bandgap at reference temperature in units of eV.
         1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
         from the SAM CEC module database, EgRef=1.121 is implicit for all
         cell types in the parameter estimation algorithm used by NREL.
    dEgdT:  The temperature dependence of the energy bandgap at reference
         conditions in units of 1/K. May be either a scalar value
         (e.g. -0.0002677 as in [1]_) or a DataFrame (this may be useful if
         dEgdT is a modeled as a function of temperature). For parameters from
         the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
         types in the parameter estimation algorithm used by NREL.

    Returns
    -------
    IV curve (pd.DataFrame):
         * i - IV curve current in amperes.
         * v - IV curve voltage in volts.

    References
    -------
    Strongly inspired/taken from: https://pvlib-python.readthedocs.io/en/v0.9.0/auto_examples/plot_singlediode.html

    """
    # adjust the reference parameters according to the operating conditions (effective_irradiance, temp_cell)
    # using the De Soto model:
    IL, I0, Rs, Rsh, nNsVth = calcparams_desoto(effective_irradiance, temp_cell,
                                                alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                                                EgRef, dEgdT)

    # Solve the single-diode equation to obtain a photovoltaic IV curve.
    curve_info = singlediode(
        photocurrent=IL,
        saturation_current=I0,
        resistance_series=Rs,
        resistance_shunt=Rsh,
        nNsVth=nNsVth,
        ivcurve_pnts=n_points,
        method='lambertw'
    )

    curve_df = pd.DataFrame(curve_info)[["v", "i"]]

    return curve_df


def get_Pmpp(iv_curve: pd.DataFrame, VI_max=False, v_col="v", i_col="i"):
    """
    Return the maximum power-point from an IV curve

    Parameters
    ----------
    iv_curve: Dataframe with the "i" and "v" points
    VI_max: also return "Vmpp" and "Impp" at the maximum power point
    v_col: Voltage column
    i_col: Current column
    """
    p = iv_curve[v_col] * iv_curve[i_col]
    Pmpp = p.max()

    if VI_max:
        I_mpp, V_mpp = iv_curve.loc[p.idxmax(), [i_col, v_col]].values
        return Pmpp, I_mpp, V_mpp
    else:
        return Pmpp


def curve_plot(iv_curve, v_col="v", i_col="i", legend: str = None, show_Pmax=True):
    """ Plot IV curve with maximum power point"""
    ax = plt.plot(iv_curve[v_col], iv_curve[i_col], label=legend)
    if show_Pmax:
        Pmax, I_max, V_max = get_Pmpp(iv_curve, VI_max=True)
        plt.plot(V_max, I_max, label=(legend + "-Pmpp" if legend is not None else "Pmpp"), color="red", marker="o")
    plt.title("VI Curve")
    plt.ylabel("Intensity [A]")
    plt.xlabel("Voltage [V]")
    plt.legend()

    return ax


if __name__ == '__main__':
    import pvlib
    from src.utils.helio_fmt import setup_helio_plt

    pv_params = pvlib.pvsystem.retrieve_sam('cecmod')['Hanwha_Q_CELLS_Q_PEAK_DUO_G5_SC_325']
    vi_curve = vi_curve_singlediode(pv_params["alpha_sc"], pv_params["a_ref"], pv_params["I_L_ref"],
                                    pv_params["I_o_ref"], pv_params["R_sh_ref"], pv_params["R_s"])

    setup_helio_plt()
    curve_plot(vi_curve)
