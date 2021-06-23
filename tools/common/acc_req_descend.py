import random
import string
import numpy as np
import matplotlib
from scipy.optimize import fsolve

inf = float('inf')


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Modify This ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# accuracy requirement decending formulla
def acc_requirement(time_s):
    # return (-0.00003) * time_s + 1
    return -(0.001 * time_s) ** 6 - (0.00003 * time_s) + 1
    # return np.full_like(time_s, 0.85, dtype=np.double)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# reversed acc_requirement function (takes acc as input)
def time_acc_pass(acc):
    def acc_req_offset(time):
        return acc_requirement(time) - acc

    reverse = fsolve(acc_req_offset, 1e-3, full_output=True)
    if reverse[2] == 1:
        # fsolve --> x --> [0]
        return reverse[0][0]
    else:
        return np.nan


# return first_hit time
def find_first_hit(times, accs):
    # always use best 'acc' so far
    accs = np.maximum.accumulate(accs)

    passed = accs > acc_requirement(times)
    idx_first_passed = np.argmax(passed)

    # if any hit is found for given epochs
    if passed.any():
        return times[idx_first_passed], accs[idx_first_passed]

    max_acc = accs[-1]
    t_delay_hit = time_acc_pass(max_acc)

    # if dedayed hit is possiable
    if t_delay_hit is np.nan:
        return t_delay_hit, max_acc

    # give up, return 'nan'
    else:
        return np.nan, np.nan


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
