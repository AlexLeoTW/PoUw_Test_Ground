import pandas as pd
import os
import sys

conv_filters = [16, 32, 64]
conv_kernel_size = [2, 3, 4]
conv_num = [2, 4]
pool = [2, 3]
stack = ['independent', '2_in_a_row']

stat_path = sys.argv[1] if len(sys.argv) >= 2 else 'statistics.csv'
if not os.path.isfile(stat_path):
    raise FileNotFoundError('file not found')
    sys.exit(1)


df = pd.read_csv(stat_path)

params = pd.DataFrame(columns=['conv_filters', 'conv_kernel_size', 'conv_num', 'pool', 'stack'])

for n_conv_filters in conv_filters:
    for n_conv_kernel_size in conv_kernel_size:
        for n_conv_num in conv_num:
            for n_pool in pool:
                for n_stack in stack:
                    param = {
                        'conv_filters': n_conv_filters,
                        'conv_kernel_size': n_conv_kernel_size,
                        'conv_num': n_conv_num,
                        'pool': n_pool,
                        'stack': n_stack
                    }
                    # print(pd.DataFrame([param.values()], columns=param.keys()))
                    params = params.append(
                        pd.DataFrame([param.values()], columns=list(param.keys())),
                        ignore_index=True)

runnable_params = df[['conv_filters', 'conv_kernel_size', 'conv_num', 'pool', 'stack']].drop_duplicates()

not_params = pd.concat([params, runnable_params]).drop_duplicates(keep=False)

print(not_params.to_markdown())
