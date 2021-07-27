from itertools import cycle
from matplotlib import patheffects as path_effects

white = (1, 1, 1, 1)
black = (0, 0, 0, 1)
deep_gray = (0.1875, 0.1875, 0.1875, 1)     #303030
lightgray = (0.8275, 0.8275, 0.8275, 1)     #d3d3d3
transparent = (1, 1, 1, 0)

facecolor = white
acc_req_line = {'c': deep_gray, 'linestyle': '--', 'linewidth': '2'}
ecolor = black

bg_dot_color = '#808080'  # 50% gray
bg_dot_alpha = 0.5  # 50% transparent

fg_dot_color = [
    '#1f77b4',  # blue (Matisse approx.)
    '#ff7f0e',  # orange (Flamenco approx.)
    '#2ca02c',  # green (Forest Green approx.)
    '#d62728',  # red (Punch approx.)
    '#9467bd',  # purple (Wisteri aapprox.)
    '#8c564b',  # brown (Spicy Mix approx.)
    '#e377c2',  # pink (Orchid approx.)
]

fg_dot_off_color = [
    '#3B5F77',  # blue
    '#CC9A6F',  # orange
    '#568F56',  # green
    '#803A3A',  # red
    '#614879',  # purple
    '#705651',  # brown
    '#BDA7B6',  # pink
]

iter_fg_dot_color = (lambda : cycle(fg_dot_color))
iter_fg_dot_off_color = (lambda : cycle(fg_dot_off_color))

white_outline = path_effects.withStroke(linewidth=4, foreground='white')


def dark_mode():
    global facecolor, acc_req_line, ecolor
    facecolor = black
    acc_req_line = {'c': lightgray, 'linestyle': '--', 'linewidth': '2'}
    ecolor = white

    from matplotlib import pyplot as plt
    plt.rcParams.update({
        'lines.color': 'white',
        'patch.edgecolor': 'white',
        'text.color': 'white',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'lightgray',
        'axes.labelcolor': 'white',
        'axes.titlecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'lightgray',
        'legend.facecolor': '#3B3B3B',
        'figure.facecolor': 'black',
        'figure.edgecolor': 'black',
        'boxplot.flierprops.markeredgecolor': 'white',
        'boxplot.boxprops.color': 'white',
        'boxplot.whiskerprops.color': 'white',
        'boxplot.capprops.color': 'white',
        'boxplot.medianprops.color': 'white'
    })
