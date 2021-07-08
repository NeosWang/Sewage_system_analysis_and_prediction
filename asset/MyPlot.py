import pandas as pd
from matplotlib import pyplot as plt

def ValueDistributionPlot(series, bins=20, color="#607c8e", **kwargs):
    series.plot.hist(grid=True, bins=bins, rwidth=0.9, color=color,**kwargs)
    plt.ylabel("count")
    plt.grid(axis="y", alpha=0.75)


def PlotMulti(data, cols=None, spacing=0.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return
    colors = getattr(getattr(plotting, "_matplotlib").style, "_get_standard_colors")(
        num_colors=len(cols)
    )

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines["right"].set_position(("axes", 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(
            ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs
        )
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax