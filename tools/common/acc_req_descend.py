import random
import string
from scipy.optimize import fsolve


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Modify This ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# accuracy requirement decending formulla
def acc_requirement(time_s):
    return (-0.00003) * time_s + 1
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# reversed acc_requirement function (takes acc as input)
def acc_requirement_reverse(acc):
    def acc_req_offset(time):
        return acc_requirement(time) - acc
    return fsolve(acc_req_offset, 0)[0]


# Generate random string
def _randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


# returns a pandas DataFrame contains
def find_first_hit(df, params):
    col_name = '{}_pass'.format(_randomString(5))

    # insert a new column named XXXXX_pass indicates a hit
    df.insert(loc=len(params), column=col_name, value=df['val_acc'] >= acc_requirement(df['end_time']))

    # keep the "first" hit record, drop the others
    first_hit = df[df[col_name]]
    first_hit = first_hit.drop_duplicates(subset=params)

    # cacultate hitting time with acc of last epoch
    delayed_hits = df.drop_duplicates(subset=params)
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