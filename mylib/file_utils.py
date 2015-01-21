import csv
import os, errno
from contextlib import contextmanager, closing

def load_csv(*filenames, **kwargs):
    delimiter = kwargs.get('delimiter', ',')
    data = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            freader = csv.reader(f, delimiter=delimiter)
            header = freader.next()
            for row in freader:
                data.append(dict(zip(header, row)))
    return data

def save_csv(filename, data, delimiter=','):
    assert all(set(x.keys()) == set(data[0].keys()) for x in data )
    with open(filename, "w") as f:
        cols = data[0].keys()
        f.write("{0}\n".format(delimiter.join(cols)))
        for d in data:
            f.write("{0}\n".format(delimiter.join(map(d.__getitem__, cols))))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def wopen(filename):
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    return open(filename, 'w')

def is_file_complete(filename, nlines):
    '''checks whether the file exists and whether is has the required num of
    lines '''
    try:
        return sum(1 for l in file(filename)) == nlines
    except IOError:
        return False

