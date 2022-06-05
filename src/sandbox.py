from utils.edf_file_reader import EDFFileReader
from utils.ploter import drawPlot, windowPlot
from utils.buffer import bufferAllData
from singalPrepocessing.signalFiltering import signalFilter
from featureExtraction.variantion import variantion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np
import json
import math

# Variables
samplingFrequency = 250
time = 25

# Load data
edfData = EDFFileReader()

patientData = dict()
patients = ['h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07',
            's01', 's02', 's03', 's04', 's05', 's06', 's07']
headers = edfData.headers

# get data
for patient in patients:
    test = edfData.loadData(patient)
    test = edfData.getData(test)
    dataTest = {}
    for index, value in enumerate(test):
        dataTest[headers[index]] = value
    patientData[patient] = dataTest

X = []
y = []
for k, v in patientData.items():
    filteredSignal = signalFilter(v, 0.5, 30)
    
    if k.find('h'):
        y.append(1)
    else:
        y.append(0)
        
sequences = bufferAllData(filteredSignal, samplingFrequency, time)
X = variantion(sequences)


neight = KNeighborsClassifier(n_neighbors=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
neight.fit(X_train, y_train)
y_pred = neight.predict(X_test)

print('ACC:', accuracy_score(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))