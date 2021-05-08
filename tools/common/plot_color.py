from itertools import cycle
from matplotlib import patheffects as path_effects

white = (1, 1, 1, 1)
deep_gray = (0.1875, 0.1875, 0.1875, 1)
transparent = (1, 1, 1, 0)

facecolor = white

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
