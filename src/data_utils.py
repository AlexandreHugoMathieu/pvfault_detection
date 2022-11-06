# Created by A. MATHIEU at 04/11/2022
import pandas as pd

from pvlib.temperature import pvsyst_cell


def get_temp_cell(temp_cell: pd.Series = None,
                  temp_air: pd.Series = None,
                  poa_global: pd.Series = None) -> pd.Series:
    """
    Return directly the temp_cell dataframe or roughly estimate it if not provided thanks to the standard
    pvsyst model from pvlib

    :param temp_cell: Cell temperature used to model Imp and Vmp to calculate their variations under shading [C]
    :param temp_air: External air temperature to use for estimating the cell temperature (if temp_cell not directly provided) [C]
    :param poa_global: Total incident irradiance [W/m2].

    :return Cell temperature [C]
    References
    ----------
    pvsyst pvlib, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.pvsyst_cell.html
    """
    if temp_cell is None:
        if temp_air is None:
            raise ValueError("Either provide directly the cell temperature or the ambient temperature to estimate it")
        else:
            temp_cell = pvsyst_cell(poa_global, temp_air)  # standard parameters are in use
    return temp_cell
