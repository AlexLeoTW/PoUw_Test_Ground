import os
import sys
import pandas as pd
import cmdargv
import contextlib


# statistics.csv should be located in the same directory with per-train log
def _remap_logpath(df, statistics_path):
    df['log_path'] = df['log_path'].apply(lambda log_path: os.path.basename(log_path))


@contextlib.contextmanager
def _open_output(path=None):
    output = open(path, 'w') if path else sys.stdout

    yield output

    if path:
        output.close()


def _main():
    options = cmdargv.parse_argv()

    statistics = pd.read_csv(options.statistics)
    _remap_logpath(statistics, options.statistics)

    with _open_output(options.output) as output:
        output.write(statistics.to_csv())


if __name__ == '__main__':
    _main()
