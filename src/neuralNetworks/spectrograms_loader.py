from keras.utils import image_dataset_from_directory

input_shape = None

def get_classes(ds):
    return ds.class_names
    
def get_input_shape(ds):
    input_shape = None
    for ims, lbs in ds.take(1):
        input_shape = ims[0].shape
        label_shape = lbs[0].shape
        print('\ninput shape:', input_shape)
        print('label shape:', label_shape)
        break
        
    return input_shape
    
def load_train_ds(image_shape, train_dataset_path, data_split_ration=0.2, seed=1337, batch_size=64):
    train_ds = image_dataset_from_directory(
        directory=train_dataset_path,
        labels='inferred',
        label_mode='binary',
        seed=seed,
        shuffle=True,
        validation_split=data_split_ration,
        subset='training',
        batch_size=batch_size,
        image_size=image_shape)
    
    return train_ds
    
def load_validation_ds(image_shape, train_dataset_path, data_split_ration=0.2, seed=1337):
    validation_ds = image_dataset_from_directory(
        directory=train_dataset_path,
        labels='inferred',
        label_mode='binary',
        seed=seed,
        shuffle=True,
        validation_split=data_split_ration,
        subset='validation',
        batch_size=1,
        image_size=image_shape)
    
    return validation_ds

def load_test_ds(image_shape, test_dataset_path, seed=1337):
    test_ds = image_dataset_from_directory(
        directory=test_dataset_path,
        labels='inferred',
        label_mode='binary',
        seed=seed,
        shuffle=True,
        batch_size=1,
        image_size=image_shape)
    
    return test_ds

def load_datasets(image_shape, train_dataset_path, test_dataset_path, data_split_ration=0.2, seed=1337, x_train_batch_size=64):
    train_ds = load_train_ds(image_shape, train_dataset_path, data_split_ration, seed, x_train_batch_size)
    validation_ds = load_validation_ds(image_shape, train_dataset_path, data_split_ration, seed)
    
    test_ds = load_test_ds(image_shape, test_dataset_path, seed)
    
    print('\nclasses:', get_classes(test_ds))

    global input_shape
    input_shape = get_input_shape(test_ds)
    
    return train_ds, validation_ds, test_ds


    
def print_samples(ds):
    plt.figure(figsize=(10, 10))
    
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(ds.class_names[int(labels[i])])
            plt.axis("off")
           


