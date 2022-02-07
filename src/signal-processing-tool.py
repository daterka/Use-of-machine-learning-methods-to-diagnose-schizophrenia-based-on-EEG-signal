import numpy as np
import scipy.signal as signal
from pyedflib import highlevel
from os import listdir, makedirs
from os.path import isfile, join, splitext, isdir


class SignalProcessingTool:
    patientData = dict()

    def __init__(self, dataFolderPath = '../data'):
        self.filesName = [splitext(file)[0] for file in listdir(dataFolderPath) if isfile(join(dataFolderPath, file))]
        self.filesList = [join(dataFolderPath, f'{file}.edf') for file in self.filesName]


    def getHeaders(self, file):
        """Return label from readed file

        Returns:
            labelList(String[]): List of electrodes used on patient
        """
        return [headerInfo['label'] for headerInfo in highlevel.read_edf(file)[1]]
    

    def createCSVFile(self, data, headersLabel, fileName):
        """Create csv file from data

        Parameters:
            data(Any): Data used to save
            headersLabel(String[]): Labels for header
            fileName(String): Name of file
        """
        import pandas as pd
        dictForCSV = dict()
        for index, column in enumerate(data):
            dictForCSV[headersLabel[index]] = column
        pd.DataFrame(dictForCSV).to_csv(f'../output/{fileName}.csv', index = False)

    
    def interactiveInput(self):
        """Interactive mode for drawing every wavelenght for any patient"""
        printStr = 'Choose one patient from below:\n'
        for index, patient in enumerate(self.filesName):
            printStr += f'{index} - {patient} '
        
        choosenPatient = int(input(printStr+'\n'))

        printStr = 'Choose one option from below:\n'
        for index, label in enumerate(self.getHeaders(self.filesList[0])):
            printStr += f'{index} - {label} '
        
        choosenOption = int(input(printStr+'\n'))
        self.drawPlot(choosenPatient, choosenOption)


    def drawPlot(self, fileName, option):
        """Draw wavelenght from edf file

        Parameters:
            fileName(String): Name of file
            option(String): Choosen wave to draw
        """
        import matplotlib.pyplot as plt
        choosenOptionLabel = self.getHeaders(self.filesList[fileName])[option]
        signals = highlevel.read_edf(self.filesList[fileName])[0]
        data = signal.resample(signals.T, 6250)

        t1 = np.arange(0, 231250, 1)
        t2 = np.arange(0, 231250, 37)

        plt.plot(t1 / 9250, signals[option], 'r', label='Before resampling')
        plt.plot(t2 / 9250, data.T[option], 'g', label='After resampling')
        plt.title(f'Patient: {self.filesName[fileName]} - Electrode: {choosenOptionLabel}')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Voltage [µV]")
        plt.show()


    def loadEDFAndSampleData(self, isFileRequired = False):
        """Draw wavelenght from edf file

        Parameters:
            isFileRequired(Boolean): Should data save to variables or files
        """
        for index, file in enumerate(self.filesList):

            if isFileRequired:
                outputFolder = '../output'
                if not isdir(outputFolder):
                    makedirs(outputFolder)
                if not isfile(join(outputFolder, f'{self.filesName[index]}.csv')):
                    signals, _, _  = highlevel.read_edf(file)
                    data = signal.resample(signals.T, 6250)
                    self.createCSVFile(data.T, self.getHeaders(self.filesList[index]), self.filesName[index])

            else:
                signals, _, _  = highlevel.read_edf(file)
                headers = np.array(self.getHeaders(self.filesList[index]))
                data = np.vstack((headers,signal.resample(signals.T, 6250)))
                self.patientData[self.filesName[index]] = data


if __name__ == "__main__":
    test = SignalProcessingTool()
    test.loadEDFAndSampleData(isFileRequired=True)