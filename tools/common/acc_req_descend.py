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
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# reversed acc_requirement function (takes acc as input)
def time_acc_pass(acc):
    def acc_req_offset(time):
        return acc_requirement(time) - acc
    return fsolve(acc_req_offset, 1e-3)[0]


# return first_hit time
def find_first_hit(times, accs):
    # always consider best 'acc' so far
    accs = np.maximum.accumulate(accs)

    # just in case a set of training can't passs acc_requirement() entire time
    max_acc = accs[-1]
    times = np.append(times, time_acc_pass(max_acc))
    accs = np.append(accs, max_acc)

    passed = accs > acc_requirement(times)
    idx_first_passed = np.argmax(passed)

    return times[idx_first_passed], accs[idx_first_passed]


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
