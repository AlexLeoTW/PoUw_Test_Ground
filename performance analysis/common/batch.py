import os
from functools import reduce

# ==============================================================================
# ||                         command line / user-inputs                       ||
# ==============================================================================
import argparse

# parse command-line arguments
def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exec', help='execute/continue batch jobs',
                        action='store_true', default=False)
    parser.add_argument('-c', '--check', help='check for output files',
                        action='store_true', default=False)
    parser.add_argument('--datadir', help='where output files stored',
                        metavar='path', default=None)

    args = parser.parse_args()

    # default to exec when nothing passed
    if not args.exec and not args.check:
        args.exec = True

    return args


# turn config['env'] into "string":python cmd (e.g. python -v)
def mk_python_cmd(config):
    python_bin = config['env']['python_bin']
    python_args = config['env']['python_args']

    return reduce(lambda a, b: f'{a} {b}', [python_bin] + python_args)

# ==============================================================================
# ||                               file process                               ||
# ==============================================================================
import yaml

# read yaml
def read_yaml(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)

    return config


# list files in specified directory (no recursive traversal)
def list_files(path=None, match=None):
    if path is None:
        path = os.getcwd()

    if match is None:
        match = (lambda a: True)

    files = os.listdir(path)
    files = list(filter(match, files))

    return files

# ==============================================================================
# ||                             batch job file                               ||
# ==============================================================================
import copy

# save batch jobs (flush everytime tris to save, just in case)
def save_batch_jobs(jobs, filename='batch_job.yaml'):
    with open(filename, 'w') as file:
        file.write(yaml.dump(jobs))


# makde a new batch job entry
def _mk_entry(cmd):
    return {'command': cmd,  # string: command to execute
            'executed': False,  # False/Date: last executed
            'error_log': None}  # False/string: error_log if not return 0


# build dict of list of job entries
def mk_jobs(jobs_config, python_cmd='python3'):
    jobs = {}

    for job_name, job in jobs_config.items():
        _mk_cmd = (lambda args: f'{python_cmd} {job["script"]} {" ".join(args)}')

        args_gen = token_list_generator(job['args'])
        entrys = [ _mk_entry(_mk_cmd(args)) for args in args_gen ]

        # duplicate entrys / loop
        loop = job.get('loop', 1)
        looped_entries = reduce(
            lambda a, b: a + b,
            [copy.deepcopy(entrys) for idx in range(loop)])

        jobs[job_name] = looped_entries

    return jobs


# create batch job file (batch_job.yaml) if not exist
def create_batch_job_if_not_exist(config):
    python_cmd = mk_python_cmd(config)
    job_file = config['env'].get('job_file', 'batch_job.yaml')

    print(f'job_file = {job_file}')

    if os.path.exists(job_file):
        print('continue batch jobs')
        jobs = read_yaml(job_file)
    else:
        print(f'"{job_file}" not found, create one')
        jobs = mk_jobs(config['jobs'], python_cmd=python_cmd)
        save_batch_jobs(jobs, job_file)

    return jobs

# ==============================================================================
# ||                           arguments / tokens                             ||
# ==============================================================================
import numpy as np

# make every elements in a list/2D-list str type
def as_str(lis):
    return np.array(lis).astype(str).tolist()

# treat arguments and its values as just tokens,
#   return: ['--foo', ['1', '2', '3'], '--ber', ['4', '5', '6']]
def args_to_token_list(args):
    tokens = []

    for arg in args:
        if isinstance(arg, str):
            tokens.append([arg])
            continue

        if isinstance(arg, int):
            tokens.append([str(arg)])
            continue

        if isinstance(arg, list):
            tokens.append(as_str(str, arg))
            continue

        # {key: value} ==> key, value
        key, values = list(arg.items())[0]
        tokens.append([key])
        # if is 2D-list
        if type(values) == list and type(values[0]) == list:
            tokens.extend(as_str(values))
        else:
            tokens.append(as_str(values))

    return tokens


# return a generator, repeatedly yield a token each time called
def _token_gen(tokens, dup=1, loop=1):
    for _l in range(loop):
        for token in tokens:
            for _d in range(dup):
                yield token


# return a generator, generates a list of command-line arguments/tokens each time
def token_list_generator(args):
    # split arguments and its variables into separate items in the list
    tokens = args_to_token_list(args)
    # how many possiable value is of each item(variable) in the list
    tokens_len = [len(token) for token in tokens]
    total_len = np.prod(tokens_len)

    # how many times a possiable token value (variable) should be repeated
    #   through entire job
    tokens_dup = [ np.prod(tokens_len[idx+1:]).astype(int) for idx in range(len(tokens)) ]
    # how many times a set of duplicated token values should be looped to cover entire job
    #   i.e. ( [val1]*tokens_dup + [val1]*tokens_dup + ... ) * tokens_loop
    tokens_loop = (total_len / tokens_dup).astype(int)

    token_gens = [ _token_gen(tokens[idx], dup=tokens_dup[idx], loop=tokens_loop[idx]) for idx in range(len(tokens)) ]

    for idx in range(total_len):
        yield [ next(token_gen) for token_gen in token_gens ]

# ==============================================================================
# ||                               shell / exec                               ||
# ==============================================================================
import sys
import subprocess
import shlex
import time

# exec the "command" and print to shell while capturing outputs
def exec_in_shell(command, encode=None):
    print('-' * 80)
    print(command)
    print('-' * 80)

    encode = encode if encode is not None else sys.getfilesystemencoding()
    process = subprocess.Popen(shlex.split(command),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = ''

    # read output until nothing left(EOF)
    for line in iter(process.stdout.readline, b''):
        line = line.decode(encode)
        sys.stdout.write(line)
        output += line

    process.wait()

    return process.returncode, output

# loop-through entries in jobs object(dict)
def run_jobs(jobs, resume=True):
    for job in jobs:
        for idx in range(len(jobs[job])):
            # skip executed entries
            if resume and jobs[job][idx]['executed']:
                continue

            code, output = exec_in_shell(jobs[job][idx]['command'])
            timestamp = time.strftime("%Y-%m-%d_%H:%M:%S UTC", time.gmtime())

            jobs[job][idx]['executed'] = timestamp

            if code == 0:
                jobs[job][idx]['error_log'] = None
            else:
                log_path = save_error_log(output, timestamp)
                jobs[job][idx]['error_log'] = log_path

            save_batch_jobs(jobs)

# save error log in current directory
def save_error_log(log, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S UTC", time.gmtime())
    log_path = f'error_log_{timestamp}.log'

    with open(log_path, 'w') as file:
        file.write(log)

    return log_path


# ==============================================================================
# ||                                   check                                  ||
# ==============================================================================
import re
from itertools import count as cnt_from
from tabulate import tabulate

# excrat items from a list by position(index)
def extract_items(items, pos):
    return [items[idx] for idx in pos]


# check if filename match (regex rules) and (token value)
def match_filename(filename, regex, tokens):
    matchs = re.match(regex, filename).groups()

    if not matchs:
        return false
    # check if captured values match
    return all([matchs[idx] == tokens[idx] for idx in range(len(tokens))])


# locate files that matchs (regex rules) and (token value), return their index
def match_in_files(files, regex, tokens):
    matched_idx = []

    for idx in range(len(files)):
        if match_filename(files[idx], regex, tokens):
            matched_idx.append(idx)

    return  matched_idx


def check_job_output(job, datadir=None):
    tokens_gen = token_list_generator(job['args'])
    check = job.get('check', None)
    loop = job.get('loop', 1)
    missing_cmds = []

    if not check:
        raise KeyError('check specified is not configured, check "batch.yaml"')

    files = list_files(datadir, match=lambda file: re.match(check['filename'], file))

    for idx, tokened_args in zip(cnt_from(0), tokens_gen):
        tokens = extract_items(tokened_args, pos=check['match'])
        regex = check['filename']

        match_idx = match_in_files(files, regex=regex, tokens=tokens)

        if len(match_idx) != loop:
            missing_cmds.append({
                'idx': idx,
                'args': ' '.join(tokened_args),
                'expect': loop,
                'found': len(match_idx)
            })

    return missing_cmds


def print_missing_cmds(missing_cmds, script_cmd=None):
    if len(missing_cmds) == 0:
        print('no missing output found')
        return

    if script_cmd:
        for missing in missing_cmds:
            arg = missing.pop('args')
            missing['cmd'] = f'{script_cmd} {arg}'

    table_columns = list(missing_cmds[0].keys())
    table_content = map(lambda x: list(x.values()), missing_cmds)
    print(tabulate(table_content, table_columns))


# ==============================================================================
# ||                                  main()                                  ||
# ==============================================================================
import pprint



pp = pprint.PrettyPrinter(indent=2)

def main():
    config = read_yaml('batch.yaml')
    argv = parse_argv()

    pp.pprint(config)

    if argv.exec:
        job_file = config['env']['job_file']
        jobs = create_batch_job_if_not_exist(config)
        run_jobs(jobs)
    else: # argv.check
        python_cmd = mk_python_cmd(config)

        for job_name, job in config['jobs'].items():
            print('=' * 80)
            print(f'job: {job_name}')
            print('-' * 80)
            missing_cmds = check_job_output(job, datadir=argv.datadir)
            print_missing_cmds(
                missing_cmds, script_cmd=f'{python_cmd} {job["script"]}')


if __name__ == '__main__':
    main()
