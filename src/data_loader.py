import numpy as np
from pyedflib import highlevel
from os import listdir
from os.path import isfile, join

def dataLoader():
    data = []
    onlyfiles = [f for f in listdir('../data') if isfile(join('../data', f))]

    # Header preparation
    signals, signal_headers, _ = highlevel.read_edf( f'../data/{onlyfiles[0]}')
    signal_headers = [s['label'] for s in signal_headers] + ['label']
    data.append(signal_headers)

    print(signals.shape) # 19 x 231250
    return np.array(data)

if __name__ == "__main__":
    print(dataLoader())