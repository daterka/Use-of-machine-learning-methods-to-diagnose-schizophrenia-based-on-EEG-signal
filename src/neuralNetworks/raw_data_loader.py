# imports
import gc
import os

import mne
from sklearn.model_selection import train_test_split

import numpy as np


input_shape = None

# load edf
def load_patients_data(edfs_path):
    raw_patients_data = []
    
    edfs_file_names = [f for f in os.listdir(edfs_path) if f.endswith('.edf')]
    
    for file_name in edfs_file_names:
        path = edfs_path + '\\' + file_name 
        raw_data = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw_patients_data.append(raw_data)

    return raw_patients_data

# signal segmentation
def get_label(edf):
    patient_edf_file_name = edf.filenames[0].split('\\')[-1]
    isSick = patient_edf_file_name.lower().startswith('s')
    return int(isSick == True) # 1 - is sick, 0 is healthy

def get_min_max_duration_for_classes(raw_patients_data, print_durations=False):
    min_SZ_negative_duration = float("inf") # healthy
    min_SZ_positive_duration = float("inf") # sick

    max_SZ_negative_duration = 0 # healthy
    max_SZ_positive_duration = 0 # sick

    for edf in raw_patients_data:
        duration = edf.times[-1]

        if(get_label(edf) == 0):
            min_SZ_negative_duration = duration if duration < min_SZ_negative_duration else min_SZ_negative_duration
            max_SZ_negative_duration = duration if duration > max_SZ_negative_duration else max_SZ_negative_duration
        else:
            min_SZ_positive_duration = duration if duration < min_SZ_positive_duration else min_SZ_positive_duration
            max_SZ_positive_duration = duration if duration > max_SZ_positive_duration else max_SZ_positive_duration


    print('SZ_negative: min =', min_SZ_negative_duration, ', max =', max_SZ_negative_duration)
    print('SZ_positive: min =', min_SZ_positive_duration, ', max =', max_SZ_positive_duration)
    
    return min_SZ_negative_duration, min_SZ_positive_duration, max_SZ_negative_duration, max_SZ_positive_duration

def crop_raw_data_to_equalize_duration_per_class(raw_patients_data):
    print("Duration per class before cropping: ")
    min_dur_neg, min_dur_pos, *_ = get_min_max_duration_for_classes(raw_patients_data, True)
    
    for edf in raw_patients_data:
        duration = edf.times[-1]

        if(get_label(edf) == 0):
            if(duration > min_dur_neg):
                edf.crop(tmin=0, tmax=min_dur_neg, include_tmax=True)
        else:
            if(duration > min_dur_pos):
                edf.crop(tmin=0, tmax=min_dur_pos, include_tmax=True)
                
    print("\nDuration per class after cropping: ")

    get_min_max_duration_for_classes(raw_patients_data, True)

def print_info(epochs_num_per_patient, labels):
    print('\nEpochs number per patient: ', epochs_num_per_patient)
    
    class_SZ_positive = sum(labels) 
    class_SZ_negative= len(labels)-sum(labels)

    print('\nnegative: ', class_SZ_positive)
    print('positive: ', class_SZ_negative)
    
def transform_patients_data_into_X_y_sets(raw_patients_data, segment_duration=5.0, info=True):
    epochs_per_patient = []
    labels = []
    
    epochs_num_per_patient = []
    for edf in raw_patients_data:
        epochs = mne.make_fixed_length_epochs(edf, duration=segment_duration, preload=True, verbose=False)
        epochs_per_patient.append(epochs)
        epochs_num_per_patient.append(len(epochs))
        
        label = get_label(edf)
        labels.extend([label for epoch in epochs])
    
    epochs = mne.concatenate_epochs(epochs_per_patient)

    if info:
        print_info(epochs_num_per_patient, labels)
        
#     del epochs_num_per_patient
#     gc.collect()
    
    return (epochs, np.array(labels)) # (X, y)

def reshape_data(X):
    X_shape = X[0].get_data().shape
    X_shape = (len(X), X_shape[2], X_shape[1])
    X_shape
    
    X_data = np.zeros(shape=X_shape)

    for i in range(len(X)):
        df = X[i].to_data_frame().drop(['time', 'condition', 'epoch'], axis=1)
        epoch_data = df.to_numpy()
        X_data[i] = epoch_data
        
    return X_data    

def load_and_split_data(edfs_path, segment_len=5.0, split_ratio=0.2, seed=1337):
    raw_patients_data = load_patients_data(edfs_path)
    
    X, y = transform_patients_data_into_X_y_sets(raw_patients_data=raw_patients_data, segment_duration=segment_len)
    
    X_data = reshape_data(X)
    
    global input_shape
    input_shape = (X_data.shape[1], X_data.shape[2]) # (5000, 19)
        
    del raw_patients_data
    gc.collect()
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=split_ratio, shuffle=True, random_state=seed)
    
    print('\nX_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    print('\ninput shape:', input_shape)
    
    return X_train, X_test, y_train, y_test

def load_data(edfs_path, segment_len=5.0, split_ratio=0.2, seed=1337):
    raw_patients_data = load_patients_data(edfs_path)
    
    X, y = transform_patients_data_into_X_y_sets(raw_patients_data=raw_patients_data, segment_duration=segment_len)
            
    del raw_patients_data
    gc.collect()
    
    return X, y
    
    
    
    
    
    