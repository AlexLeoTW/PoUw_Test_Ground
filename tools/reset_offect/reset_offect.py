import os
import sys
import pandas as pd

path = sys.argv[1]
dirname = os.path.dirname(path)

statistics = pd.read_csv(path)
log_path_s = statistics['log_path'].apply(lambda x: os.path.join(dirname, x))

for index, log_path in log_path_s.items():
    log = pd.read_csv(log_path)
    start_time = log.loc[0, 'start_time']
    log['start_time'] = log.loc[:, 'start_time'].apply(lambda x: x - start_time)
    log['end_time'] = log.loc[:, 'end_time'].apply(lambda x: x - start_time)

    new_log_path = f'{log_path[:-4]}_fix.csv'
    print(f'write file "{new_log_path}"')
    with open(new_log_path, 'w') as file:
        file.write(log.to_csv())

statistics['log_path'] = statistics.loc[:, 'log_path'].apply(lambda log_path: f'{log_path[:-4]}_fix.csv', 'w')

new_path = f'{path[:-4]}_fix.csv'
print(f'write file "{new_path}"')
with open(new_path, 'w') as file:
    file.write(statistics.to_csv())
