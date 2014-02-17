# from pandas.compat import range, lrange, lmap, map, zip
from pandas.tools.plotting import _get_standard_colors
import pandas.core.common as com

def andrews_curves(data, class_column, ax=None, samples=200, colormap=None,
                   **kwds):
    """
    Parameters:
    -----------
    data : DataFrame
        Data to be plotted, preferably normalized to (0.0, 1.0)
    class_column : Name of the column containing class names
    ax : matplotlib axes object, default None
    samples : Number of points to plot in each curve
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that name
        from matplotlib.
    kwds : Optional plotting arguments to be passed to matplotlib

    Returns:
    --------
    ax: Matplotlib axis object

    """
    from math import sqrt, pi, sin, cos
    import matplotlib.pyplot as plt

    def function(amplitudes):
        def f(x):
            x1 = amplitudes[0]
            result = x1 / sqrt(2.0)
            harmonic = 1.0
            for x_even, x_odd in zip(amplitudes[1::2], amplitudes[2::2]):
                result += (x_even * sin(harmonic * x) +
                           x_odd * cos(harmonic * x))
                harmonic += 1.0
            if len(amplitudes) % 2 != 0:
                result += amplitudes[-1] * sin(harmonic * x)
            return result
        return f

    n = len(data)
    class_col = data[class_column]
    uniq_class = class_col.drop_duplicates()
    columns = [data[col] for col in data.columns if (col != class_column)]
    x = [-pi + 2.0 * pi * (t / float(samples)) for t in range(samples)]
    used_legends = set([])

    colors = _get_standard_colors(num_colors=len(uniq_class), colormap=colormap,
                                  color_type='random', color=kwds.get('color'))
    col_dict = dict([(klass, col) for klass, col in zip(uniq_class, colors)])
    if ax is None:
        ax = plt.gca(xlim=(-pi, pi))
    for i in range(n):
        row = [columns[c].iloc[i] for c in range(len(columns))]
        f = function(row)
        y = [f(t) for t in x]
        label = None
        if com.pprint_thing(class_col.iloc[i]) not in used_legends:
            label = com.pprint_thing(class_col.iloc[i])
            used_legends.add(label)
            ax.plot(x, y, color=col_dict[class_col.iloc[i]], label=label, **kwds)
        else:
            ax.plot(x, y, color=col_dict[class_col.iloc[i]], **kwds)

    ax.legend(loc='upper right')
    ax.grid()
    return ax