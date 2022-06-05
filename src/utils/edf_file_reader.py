import numpy as np
import scipy.signal as signal
from pyedflib import highlevel
from os import listdir, makedirs
from os.path import isfile, join, splitext, isdir, abspath


class EDFFileReader:
    absolutePath = abspath('.').replace('/src', '').replace('/utils', '')

    def __init__(self):
        self.files = {file.split('.')[0]: join(self.absolutePath, 'data', file) for file in listdir(join(self.absolutePath, 'data'))}
        self.headers = [headerInfo['label'] for headerInfo in highlevel.read_edf(self.files['h01'])[1]]

    def loadData(self, fileName):
        return highlevel.read_edf(self.files[fileName])

    def getData(self, fileContent):
        return np.array([signals for signals in fileContent[0]])

    def getAllData(self, toList = False, onlyShapes = False):
        patientDataSet = {}
        for file in self.files.keys():
            fileContent = self.loadData(file)
            patientDataSet[file] = {}
            
            for electrode in self.getHeaders(fileContent):
                if toList:
                    for item in self.getData(fileContent):
                        patientDataSet[file][electrode] = item.tolist()
                elif onlyShapes:
                    patientDataSet[file][electrode] = self.getData(fileContent).shape
                else:
                    patientDataSet[file][electrode] = self.getData(fileContent)

        return patientDataSet

    def createCSVFile(self, fileName):
        outputAbsPath = join(self.absolutePath, 'output')
        if not isdir(outputAbsPath):
            makedirs(outputAbsPath)

        import pandas as pd
        dictForCSV = dict()

        fileContent = self.loadData(fileName)
        headers = self.getHeaders(fileContent)
        for index, column in enumerate(self.getData(fileContent)):
            dictForCSV[headers[index]] = column
        pd.DataFrame(dictForCSV).to_csv(join(outputAbsPath, f'{fileName}.csv'), index = False)


if __name__ == "__main__":
    test = EDFFileReader()
    test.createCSVFile('h01')