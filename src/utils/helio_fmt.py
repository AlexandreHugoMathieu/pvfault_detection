""" This modules contains all functions to format objects with Heliocity visual chart"""
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

# Heliocity chart-official colors
orange = (243 / 255, 146 / 255, 0 / 255)
grey = (87 / 255, 87 / 255, 86 / 255)

# Non official colors
blue = (40 / 255, 106 / 255, 162 / 255)
black = (46 / 255, 54 / 255, 71 / 255)
yellow = (229 / 255, 199 / 255, 34 / 255)

# Extra non official colors
red = (217 / 255, 25 / 255, 38 / 255)
bluedark = (31 / 255, 80 / 255, 125 / 255)

# Colors implemented by default in matplotlib cycle
COLORS = [orange, grey, yellow, black, blue]


def setup_helio_plt(font: dict = {'family': 'Calibri', 'size': 12}) -> None:
    """
    Set up the format of matplotlib plots to Heliocity Format.

    :param font: dictionary with font caracteristics
    """
    mpl.rcParams['axes.prop_cycle'] = cycler(color=COLORS)
    mpl.rc('font', **font)
    mpl.rcParams['lines.linewidth'] = 1.5
    return None


def random_plot() -> plt.Figure:
    """ Generates a random plot with helio plot format"""
    setup_helio_plt()
    df = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'))
    df.plot.bar()


if __name__ == "__main__":
    random_plot()
