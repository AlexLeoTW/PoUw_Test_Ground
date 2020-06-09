import random
import string
import numpy as np
from scipy.optimize import fsolve

inf = float('inf')


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Modify This ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# accuracy requirement decending formulla
def acc_requirement(time_s):
    # return (-0.00003) * time_s + 1
    return -(0.001 * time_s) ** 6 - (0.00003 * time_s) + 1
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# reversed acc_requirement function (takes acc as input)
def acc_requirement_reverse(acc):
    def acc_req_offset(time):
        return acc_requirement(time) - acc
    return fsolve(acc_req_offset, 1e-3)[0]


# Generate random string
def _randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


# returns a pandas DataFrame contains
def find_first_hit(df, params):
    col_name = '{}_pass'.format(_randomString(5))

    # insert a new column named XXXXX_pass indicates a hit
    df.insert(loc=len(params), column=col_name,
              value=df['val_acc'] >= acc_requirement(df['end_time']))

    # keep the "first" hit record, drop the others
    first_hit = df[df[col_name]]
    first_hit = first_hit.drop_duplicates(subset=params, keep='first')

    # cacultate hitting time with acc of last epoch
    delayed_hits = df.copy()
    delayed_hits = delayed_hits.sort_values(by='val_acc', ascending=False)  # descending
    delayed_hits = delayed_hits.drop_duplicates(subset=params, keep='first')
    def update_val_acc(col):
        col['end_time'] = acc_requirement_reverse(col['val_acc'])
        return col
    delayed_hits = delayed_hits.apply(update_val_acc, axis='columns')

    # fill missing hits with delayed_hits
    first_hit = first_hit.append(delayed_hits)
    first_hit = first_hit.drop_duplicates(subset=params, keep='first')
    first_hit = first_hit.sort_values(by=params)
    first_hit.reset_index(drop=True, inplace=True)

    # delete(drop) XXXXX_pass column
    df.drop(columns=[col_name], inplace=True)
    first_hit.drop(columns=[col_name], inplace=True)

    return first_hit


def draw_decending_acc_requirement(plt, **kwargs):
    x_min, x_max = plt.xlim()
    x_min = max(x_min, 0)

    xs = np.arange(x_min, x_max, 5)
    ys = acc_requirement(xs)

    default_plot_arg = {'c': '#303030', 'linestyle': '--', 'linewidth': '2'}
    plt.plot(xs, ys, **{**default_plot_arg, **kwargs})


def x_y_lim(first_hits, margin=0, expand=False,
            fit_curve=False, clip=[(-inf, -inf), (inf, inf)]):
    # first_hits: pd.DataFrame / assume {*params, val_acc, end_time}
    # margin: *fraction* of the span between min and max
    y_max, x_max = first_hits[['val_acc', 'end_time']].max(axis='index').values
    y_min, x_min = first_hits[['val_acc', 'end_time']].min(axis='index').values

    if expand:
        x_min, y_min = 0, 0
        y_max = 1

    if fit_curve:
        # might be > 1, when curve(0)
        y_max = max(min(acc_requirement(0), 1), y_max)
        x_max = max(acc_requirement_reverse(0), x_max)

    if margin:
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin

        x_min, x_max = x_min - x_margin, x_max + x_margin
        y_min, y_max = y_min - y_margin, y_max + y_margin

    x_min = x_min if x_min > clip[0][0] else clip[0][0]
    y_min = y_min if y_min > clip[0][1] else clip[0][1]
    x_max = x_max if x_max < clip[1][0] else clip[1][0]
    y_max = y_max if y_max < clip[1][1] else clip[1][1]

    return (x_min, x_max), (y_min, y_max)  # x_lim, y_lim
