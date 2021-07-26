import random
import string
import numpy as np
import matplotlib
from scipy.optimize import fsolve

inf = float('inf')


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Modify This ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# accuracy requirement decending formulla
def acc_requirement(time_s):
    # return np.full_like(time_s, 0.7, dtype=np.double)         # flat
    # return (-0.003) * (time_s - 480)  + 1                     # straight
    # return -(0.001 * time_s) ** 6 - (0.00003 * time_s) + 1    # curve
    return - 1.02 ** (time_s - 600) + 1                         # curve2


def reverse_acc_req(acc):
    # return None                                               # flat
    # return (acc - 1) * -1000 / 3 + 480                        # flat
    # return None                                               # curve
    return np.log(1 - acc) / np.log(1.02) + 600                 # curve2
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# return start_time and end_time of the acc-req. plot
def get_x_range():
    end_time = time_acc_pass(acc=0)
    return (0, end_time)


# return line (xs, ys) in "x_range" and "y_range"
def get_acc_req_xy(x_range=None, y_range=(0, 1)):
    x_min, x_max = x_range if x_range else get_x_range()
    y_min, y_max = y_range

    # just in case
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    y_max = min(y_max, 1)

    xs = np.arange(x_min, x_max + 1, 5)  # "x_max + 1" is to make sure to cover x_max
    ys = acc_requirement(xs)

    # cut out of range (e.g. acc=1)
    keep = np.logical_and(ys >= y_min, ys <= y_max)
    xs = xs[keep]
    ys = ys[keep]

    return xs, ys
