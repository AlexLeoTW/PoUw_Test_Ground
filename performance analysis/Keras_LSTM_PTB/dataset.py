import os
import shutil
import urllib.request
import sklearn.preprocessing

data_path = os.path.join(os.path.dirname(__file__), 'dataset')


def _read_words(filename):
    with open(filename, 'r') as file:
        return file.read().replace('\n', '<eos>').split()


def _check_ptb():
    if not os.path.isdir(data_path):
        return False
    for file in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']:
        if not os.path.isfile(os.path.join(data_path, file)):
            return False

    return True


def _download_ptb():
    shutil.rmtree(data_path)
    os.mkdir(data_path)

    for file in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']:
        print('Downloading {file}...'.format(file=file))
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{file}'.format(file=file), filename=os.path.join(data_path, file))


def load_ptb():
    # if not local copy, download it
    if not _check_ptb():
        _download_ptb()

    # get the data paths
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    # read files
    train_data = _read_words(train_path)
    valid_data = _read_words(valid_path)
    test_data = _read_words(test_path)

    # build the complete num_vocabulary
    labelEnc = sklearn.preprocessing.LabelEncoder()
    labelEnc.fit(train_data)

    # convert text data to list of integers
    train_data = labelEnc.transform(train_data)
    valid_data = labelEnc.transform(valid_data)
    test_data = labelEnc.transform(test_data)

    return train_data, valid_data, test_data, labelEnc


def main():
    load_ptb()


if __name__ == '__main__':
    main()
