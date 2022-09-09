import matplotlib.pyplot as plt
import numpy as np

def drawPlot(dataSet, electrode, patient, newSampling = [1]):

    for index, data in enumerate(dataSet):
        t = np.arange(0, data.shape[0] / newSampling[index], 1 / newSampling[index])
        plt.plot(t, data)


    plt.title(f'Patient: {patient} - Electrode: {electrode}')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [µV]")
    plt.grid()
    plt.show()

def windowPlot(dataSet, electrode, patient):
    start = 0
    for index, data in enumerate(dataSet):
        t = np.arange((index * start) / 250, (index * start + data.shape[0]) / 250, 1 / 250)
        start = data.shape[0]
        plt.plot(t, data)


    plt.title(f'Patient: {patient} - Electrode: {electrode}')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [µV]")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    from edf_file_reader import EDFFileReader
    test = EDFFileReader()
    drawPlot([test.getData('h01')[0]], test.getHeaders('h01')[0], 'h01')
    