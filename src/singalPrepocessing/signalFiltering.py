from scipy.signal import butter, lfilter
import numpy as np

def signalFilter(dataSet, minFreq, maxFreq):
    b, a = butter(2, [minFreq, maxFreq], btype='band', fs=250)
    filtredDataSet = {}
    for electrode, data in dataSet.items():
        filtredDataSet[electrode] = np.array(lfilter(b, a, data))

    return filtredDataSet