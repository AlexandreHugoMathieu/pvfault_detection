# Created by A. MATHIEU at 20/10/2022
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def cldr_heatmap(df_input: pd.DataFrame,
                 col_name: str,
                 figure_title: str = "",
                 zmax: float = None,
                 zmin: float = None,
                 colormap: str = "seismic"):
    """
    Plot a pcolormesh plot with date in X axis and Time in Y axis. The color equals the value of each point.

    :param df_input: pd.DataFrame containing the data to plot
    :param col_name: Which column to plot
    :param figure_name: Name of the figure to save
    :param zmax: Maximum value to include in the visualization
    :param zmin: Minimum value to include in the visualization
    :param colormap: colormap object

    :return: figure
    """
    # Prepare plot data
    if type(col_name) is str:
        col_name = [col_name]
    df = df_input[col_name].copy()

    # Add datetime elements and make sure there is no duplicate
    df['date'] = df.index.date
    df['time'] = df.index.time
    df = df.tz_localize(None)
    df = df[~df.index.duplicated()]

    # Create a df with date and time as axis and column values as values
    pdata = df.pivot(index='date', columns='time', values=col_name)

    # Prepare labels for x and y axes as well as zvalues.
    xlabs = pdata.index.tolist()  # assume one date per value
    xaxis = np.arange(0, len(xlabs))
    step_x = max(len(pdata) // 20, 1)
    xlabs = ["" if i % step_x else xlab for i, xlab in enumerate(xlabs)]
    ylabs = list(pdata.columns.get_level_values(1))
    yaxis = np.arange(0, len(ylabs))
    step_y = len(ylabs) // 12  # xlabel every two hours
    ylabs = ["" if i % step_y else ylab.strftime("%H:%M") for i, ylab in enumerate(ylabs)]
    zmax = df[col_name].max() if zmax is None else zmax
    zmin = df[col_name].min() if zmin is None else zmin

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 6), facecolor='w')

    im = ax.imshow(pdata.T, colormap, vmin=zmin, vmax=zmax, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(xaxis, xlabs, rotation=30)
    ax.set_yticks(yaxis, ylabs)
    ax.set_facecolor('xkcd:grey')

    # Figure formating
    fig.suptitle(figure_title)
    fig.autofmt_xdate()

    return fig
