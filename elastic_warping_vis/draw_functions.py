from typing import Tuple

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.animation import FuncAnimation, FFMpegWriter

from aeon.distances import cost_matrix
from aeon.distances._alignment_paths import compute_min_return_path

from elastic_warping_vis.utils import alignment_path_to_plot


def draw_elastic(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str = "./",
    figsize: Tuple[int, int] = None,
    metric: str = "dtw",
    metric_params: dict = None,
    show_warping_connections: bool = False,
    fontsize: int = 10,
):
    """
    Draws an elastic plot comparing two time series with an elastic measure.

    Parameters
    ----------
    x : np.ndarray
        The first time series to compare. Expected to be a 2D array with shape (channels, length).
    y : np.ndarray
        The second time series to compare. Expected to be a 2D array with shape (channels, length).
    output_dir : str, optional
        The directory where the output plot will be saved. Default is "./".
    figsize : Tuple[int, int], optional
        The size of the figure in inches (width, height). Default is None, which uses a default size.
    metric : str, optional
        The metric used for DTW calculation. Default is "dtw".
    metric_params : dict, optional
        Additional parameters for the elastic metric. Default is None.
    show_warping_connections : bool, optional
        Whether to show the warping connections between the time series. Default is False.
    fontsize : int, optional
        The font size used for the plot title and labels. Default is 10.

    Returns
    -------
    None

    Saves:
    ------
    A plot comparing the two time series with elastic alignment.
    """
    n_channels = int(x.shape[0])

    blue_shades = [(0, 0, i / n_channels) for i in range(1, 1 + n_channels)]
    red_shades = [(i / n_channels, 0, 0) for i in range(1, 1 + n_channels)]

    figsize = (10, 10) if figsize is not None else figsize

    _x = np.copy(x)
    _y = np.copy(y)

    if metric_params is not None:
        _cost_matrix = cost_matrix(x=_x, y=_y, metric=metric, **metric_params)
    else:
        _cost_matrix = cost_matrix(x=_x, y=_y, metric=metric)
    optimal_path = compute_min_return_path(cost_matrix=_cost_matrix)
    path_dtw_x, path_dtw_y = alignment_path_to_plot(path_dtw=optimal_path)

    fig, ax_matrix = plt.subplots(figsize=figsize)
    fig.subplots_adjust(0.05, 0.1, 0.95, 0.95)

    divider = make_axes_locatable(ax_matrix)
    ax_x = divider.append_axes("left", size="30%", pad=0.5)
    ax_y = divider.append_axes("top", size="30%", pad=0.25)
    ax_cbar = fig.add_axes([0.82, 0.2, 0.2, 0.5])
    ax_legend = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    ax_legend.axis("off")

    lines_for_legend = []

    for c in range(n_channels):
        if c == n_channels - 1:
            lines_for_legend.append(
                ax_x.plot(
                    -x[c, 0 : len(x[c])][::-1],
                    np.arange(0, len(x[c])),
                    lw=4,
                    color=blue_shades[c],
                    label="Time Series 1",
                )[0]
            )
        else:
            lines_for_legend.append(
                ax_x.plot(
                    -x[c, 0 : len(x[c])][::-1],
                    np.arange(0, len(x[c])),
                    lw=4,
                    color=blue_shades[c],
                )[0]
            )

    for c in range(n_channels):
        if c == n_channels - 1:
            lines_for_legend.append(
                ax_y.plot(
                    np.arange(0, len(y[c])),
                    y[c, 0 : len(y[c])],
                    lw=4,
                    color=red_shades[c],
                    label="Time Series 2",
                )[0]
            )
        else:
            lines_for_legend.append(
                ax_y.plot(
                    np.arange(0, len(y[c])),
                    y[c, 0 : len(y[c])],
                    lw=4,
                    color=red_shades[c],
                )[0]
            )

    ax_x.arrow(
        x=-np.max(x) - 0.5,
        y=len(x[0]) - 1,
        dx=0,
        dy=-len(x[0]) + 1,
        head_width=0.1,
        color="gray",
    )
    ax_x.text(x=-np.max(x) - 0.7, y=0, s="time", rotation="vertical")

    ax_y.arrow(
        x=0,
        y=np.max(y) + 0.5,
        dx=len(y[0]) - 1,
        dy=0,
        head_width=0.1,
        color="gray",
    )
    ax_y.text(
        x=len(y[0]) - 1,
        y=np.max(y) + 0.7,
        s="time",
        rotation="horizontal",
    )

    im = ax_matrix.imshow(_cost_matrix, aspect="equal", cmap="cool", alpha=0.6)
    cbar = ax_cbar.figure.colorbar(im, ax=ax_cbar)
    cbar.set_label(label="Cost Matrix", size=15)

    ax_matrix.plot(path_dtw_y, path_dtw_x, color="black", lw=4)

    if show_warping_connections:
        for pair in optimal_path:

            i, j = pair[0], pair[1]
            i_mid, j_mid = i, j

            con_x = ConnectionPatch(
                (
                    -x[0, ::-1][len(x[0]) - 1 - i_mid],
                    len(x[0]) - 1 - i_mid,
                ),
                (j, i),
                "data",
                "data",
                axesA=ax_x,
                axesB=ax_matrix,
                color="orange",
                linewidth=1,
            )

            con_y = ConnectionPatch(
                (j_mid, y[0, j_mid]),
                (j, i),
                "data",
                "data",
                axesA=ax_y,
                axesB=ax_matrix,
                color="orange",
                linewidth=1,
            )

            ax_matrix.add_artist(con_x)
            ax_matrix.add_artist(con_y)

    ax_x.axis("OFF")
    ax_y.axis("OFF")
    ax_cbar.axis("OFF")

    labels = [line.get_label() for line in lines_for_legend]
    lines_for_legend.append(Line2D([], [], lw=4, color="black"))
    labels.append("Warping Path - " + metric)
    ax_legend.legend(
        lines_for_legend,
        labels,
        loc="upper left",
        prop={"size": fontsize},
        bbox_to_anchor=(-0.3, 1),
    )

    ax_x.margins(y=0)
    ax_y.margins(x=0)

    fig.savefig(os.path.join(output_dir, metric + ".pdf"), bbox_inches="tight")


def draw_elastic_gif(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str = "./",
    figsize: Tuple[int, int] = None,
    metric: str = "dtw",
    fontsize: int = 10,
    metric_params: dict = None,
    fps: int = 10,
):
    """
    Creates and saves an animated video showing the warping alignment between two time series.

    Parameters
    ----------
    x : np.ndarray
        The first time series to compare. Expected to be a 2D array with shape (channels, length).
    y : np.ndarray
        The second time series to compare. Expected to be a 2D array with shape (channels, length).
    output_dir : str, optional
        The directory where the output GIF will be saved. Default is './'.
    figsize : Tuple[int, int], optional
        The size of the figure in inches (width, height). Default is None, which uses a default size of (10, 10).
    metric : str, optional
        The metric used for DTW calculation. Default is "dtw".
    fontsize : int, optional
        The font size used for the plot title and labels. Default is 10.
    metric_params : dict, optional
        Additional parameters for the elastic metric. Default is None.
    fps: int, optional
        The number of frames per second, default is 10.

    Returns
    -------
    None

    Saves
    -----
    An animated MP$ showing the alignment of the two time series.
    """
    if figsize is None:
        figsize = (10, 10)

    n_channels = int(x.shape[0])

    blue_shades = [(0, 0, i / n_channels) for i in range(1, 1 + n_channels)]
    red_shades = [(i / n_channels, 0, 0) for i in range(1, 1 + n_channels)]

    _x = np.copy(x)
    _y = np.copy(y)

    if metric_params is not None:
        _cost_matrix = cost_matrix(x=_x, y=_y, metric=metric, **metric_params)
    else:
        _cost_matrix = cost_matrix(x=_x, y=_y, metric=metric)
    optimal_path = compute_min_return_path(cost_matrix=_cost_matrix)
    path_dtw_x, path_dtw_y = alignment_path_to_plot(path_dtw=optimal_path)

    fig, ax_matrix = plt.subplots(figsize=figsize)
    fig.subplots_adjust(0.05, 0.1, 0.95, 0.95)

    divider = make_axes_locatable(ax_matrix)
    ax_x = divider.append_axes("left", size="25%", pad=0.5)
    ax_y = divider.append_axes("top", size="25%", pad=0.25)
    ax_cbar = fig.add_axes([0.75, 0.2, 0.2, 0.5])
    ax_legend = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    ax_legend.axis("off")

    lines_for_legend = []

    for c in range(n_channels):
        if c == n_channels - 1:
            lines_for_legend.append(
                ax_x.plot(
                    -x[c, 0 : len(x[c])][::-1],
                    np.arange(0, len(x[c])),
                    lw=4,
                    color=blue_shades[c],
                    label="Time Series 1",
                )[0]
            )
        else:
            lines_for_legend.append(
                ax_x.plot(
                    -x[c, 0 : len(x[c])][::-1],
                    np.arange(0, len(x[c])),
                    lw=4,
                    color=blue_shades[c],
                )[0]
            )

    for c in range(n_channels):
        if c == n_channels - 1:
            lines_for_legend.append(
                ax_y.plot(
                    np.arange(0, len(y[c])),
                    y[c, 0 : len(y[c])],
                    lw=4,
                    color=red_shades[c],
                    label="Time Series 2",
                )[0]
            )
        else:
            lines_for_legend.append(
                ax_y.plot(
                    np.arange(0, len(y[c])),
                    y[c, 0 : len(y[c])],
                    lw=4,
                    color=red_shades[c],
                )[0]
            )

    ax_x.arrow(
        x=-np.max(x) - 0.5,
        y=len(x[0]) - 1,
        dx=0,
        dy=-len(x[0]) + 1,
        head_width=0.1,
        color="gray",
    )
    ax_x.text(x=-np.max(x) - 0.7, y=0, s="time", rotation="vertical")

    ax_y.arrow(
        x=0,
        y=np.max(y) + 0.5,
        dx=len(y[0]) - 1,
        dy=0,
        head_width=0.1,
        color="gray",
    )
    ax_y.text(
        x=len(y[0]) - 1,
        y=np.max(y) + 0.7,
        s="time",
        rotation="horizontal",
    )

    im = ax_matrix.imshow(_cost_matrix, aspect="equal", cmap="cool", alpha=0.6)

    cbar = ax_cbar.figure.colorbar(im, ax=ax_cbar)
    cbar.set_label(label="Cost Matrix", size=15)

    dtw_plot = ax_matrix.plot(path_dtw_y, path_dtw_x, color="black", lw=4)[0]

    if n_channels > 1:
        vert_x = ax_x.plot(
            [
                -x[:, ::-1][:, len(x[0]) - 1 - optimal_path[0][0]].max(),
                -x[:, ::-1][:, len(x[0]) - 1 - optimal_path[0][0]].min(),
            ],
            [len(x[0]) - 1 - optimal_path[0][0], len(x[0]) - 1 - optimal_path[0][0]],
            lw=4,
            color="orange",
            zorder=2,
        )[0]

        vert_y = ax_y.plot(
            [optimal_path[0][1], optimal_path[0][1]],
            [y[:, optimal_path[0][1]].min(), y[:, optimal_path[0][1]].max()],
            lw=4,
            color="orange",
            zorder=2,
        )[0]
    else:
        vert_x = ax_x.scatter(
            -x[0, ::-1][len(x[0]) - 1 - optimal_path[0][0]],
            len(x[0]) - 1 - optimal_path[0][0],
            s=100,
            color="orange",
            zorder=2,
        )

        vert_y = ax_y.scatter(
            optimal_path[0][1],
            y[0, optimal_path[0][1]],
            s=100,
            color="orange",
            zorder=2,
        )

    con_x = ConnectionPatch(
        (
            -x[:, ::-1][:, len(x[0]) - 1 - optimal_path[0][0]].max(),
            len(x[0]) - 1 - optimal_path[0][0],
        ),
        (optimal_path[0][1], optimal_path[0][0]),
        "data",
        "data",
        axesA=ax_x,
        axesB=ax_matrix,
        color="orange",
        linewidth=2,
    )

    con_y = ConnectionPatch(
        (optimal_path[0][1], y[:, optimal_path[0][1]].min()),
        (optimal_path[0][1], optimal_path[0][0]),
        "data",
        "data",
        axesA=ax_y,
        axesB=ax_matrix,
        color="orange",
        linewidth=2,
    )

    ax_matrix.add_artist(con_x)
    ax_matrix.add_artist(con_y)

    def animate(i):

        if i >= len(optimal_path):
            i = len(optimal_path) - 1

        time_mesh = np.arange(i + 1)

        i_x = path_dtw_x[i]
        i_y = path_dtw_y[i]
        i_x_mid, i_y_mid = i_x, i_y

        dtw_plot.set_data(path_dtw_y[time_mesh], path_dtw_x[time_mesh])

        if n_channels > 1:
            vert_x.set_data(
                [
                    -x[:, ::-1][:, len(x[0]) - 1 - i_x_mid].max(),
                    -x[:, ::-1][:, len(x[0]) - 1 - i_x_mid].min(),
                ],
                [len(x[0]) - 1 - i_x_mid, len(x[0]) - 1 - i_x_mid],
            )

            vert_y.set_data(
                [i_y_mid, i_y_mid],
                [y[:, i_y_mid].min(), y[:, i_y_mid].max()],
            )
        else:
            vert_x.set_offsets(
                [-x[0, ::-1][len(x[0]) - 1 - i_x_mid], len(x[0]) - 1 - i_x_mid]
            )

            vert_y.set_offsets([i_y_mid, y[0, i_y_mid]])

        con_x.xy1 = (
            -x[:, ::-1][:, len(x[0]) - 1 - i_x_mid].max(),
            len(x[0]) - 1 - i_x_mid,
        )
        con_x.xy2 = path_dtw_y[i], path_dtw_x[i]

        con_y.xy1 = i_y_mid, y[:, i_y_mid].min()
        con_y.xy2 = path_dtw_y[i], path_dtw_x[i]

        return dtw_plot, con_x, con_y

    ax_x.margins(y=0)
    ax_y.margins(x=0)

    ani = FuncAnimation(
        fig,
        animate,
        interval=fps,
        blit=False,
        frames=len(optimal_path) + 10,
    )

    ax_x.axis("OFF")
    ax_y.axis("OFF")
    ax_cbar.axis("OFF")

    labels = [line.get_label() for line in lines_for_legend]
    lines_for_legend.append(Line2D([], [], lw=4, color="black"))
    labels.append("Warping Path - " + metric)
    ax_legend.legend(
        lines_for_legend,
        labels,
        loc="upper left",
        prop={"size": fontsize},
        bbox_to_anchor=(-0.3, 1),
    )

    writervideo = FFMpegWriter(fps=fps)
    ani.save(os.path.join(output_dir, metric + ".mp4"), writer=writervideo)
