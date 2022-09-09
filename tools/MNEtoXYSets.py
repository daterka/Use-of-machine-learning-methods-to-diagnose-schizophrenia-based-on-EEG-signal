import mne

def get_label(edf, edf_file = None):
    if edf_file == None:
        patient_edf_file_name = edf.filenames[0].split('\\')[-1]
        isSick = patient_edf_file_name.lower().startswith('s')
    else:
        patient_edf_file_name = edf_file.split('\\')[-1]
        isSick = patient_edf_file_name.lower().startswith('s')
    return int(isSick == True) # 1 - is sick, 0 is healthy

def print_info(epochs_num_per_patient, labels):
    print('\nEpochs number per patient: ', epochs_num_per_patient)
    
    class_0_num = sum(labels) 
    class_1_num = len(labels)-sum(labels)

    print('\nnegative: ', class_0_num)
    print('positive: ', class_1_num)

def transform_patients_data_into_X_y_sets(patients_data, info=True):
    epochs_per_patient = []
    labels = []
    
    epochs_num_per_patient = []
    for edf in patients_data:
        epochs = mne.make_fixed_length_epochs(edf, duration=5, preload=True, verbose=False)
        
        epochs_per_patient.append(epochs)
        epochs_num_per_patient.append(len(epochs))
        
        label = get_label(edf)
        labels.extend([label for _ in epochs])
    
    epochs = mne.concatenate_epochs(epochs_per_patient)
    
    if info:
        print_info(epochs_num_per_patient, labels)

def transform(patients_data, info=False, edf_file=[]):
    epochs_per_patient = []
    labels = []

    for index, edf in enumerate(patients_data):
        epochs = mne.make_fixed_length_epochs(edf, duration=5, preload=True, verbose=False)
        
        epochs_per_patient.append(epochs)
        
        if len(edf_file) != 0:
            label = get_label(edf, edf_file[index])
        else:
            label = get_label(edf)

        labels.extend([label for _ in epochs])
    
    epochs = mne.concatenate_epochs(epochs_per_patient)
    
    if info:
        print_info(len(epochs), labels)

    return (epochs, labels)