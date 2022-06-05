import numpy as np
import math

def variantion(dataSet):
    featureExtractionDataSet = []

    for patient in dataSet:
        featureExtractionDataSet.append(np.log(np.var(patient, axis=1)))

    return np.array(featureExtractionDataSet)