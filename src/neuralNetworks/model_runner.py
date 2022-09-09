# imports
import gc
import os

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay, f1_score

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

# plotting results
def plot_loss(history, save_path, model_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    if len(save_path) > 0:
        plt.savefig(f'{save_path}/{model_name}-train-val-loss.png', facecolor='white', transparent=False, dpi=500)
        
    plt.show()


    
def plot_accuracy(history, save_path, model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    if len(save_path) > 0:
        plt.savefig(f'{save_path}/{model_name}-train-val-accuracy.png', facecolor='white', transparent=False, dpi=500)
    
    plt.show()
    
def print_parameters(batch_size, lr, epochs):
    print('batch size:', batch_size)
    print('learning rate:', lr)
    print('epochs:', epochs)
    
def get_batch_size(ds):
    for x, y in ds:
        return x.shape[0]
    
# EarlyStopping training
def early_stop(patience):
    return EarlyStopping(monitor='val_accuracy',
                           patience=patience,
                           restore_best_weights=True,
                           mode='max', 
                           verbose=1)

# ModelCheckpoint callback - save best weights
def model_checkpoint(save_path, model_name):
    weights_path = '{0}models/{1}.weights.best.hdf5'.format(save_path, model_name)
    return ModelCheckpoint(filepath=weights_path,
                           save_best_only=True,
                           verbose=1)

# evalute model
def test_model(test_ds, model, save_path):
    predictions = np.array([])
    labels =  np.array([])

    for x, y in test_ds:
        predictions = np.concatenate([predictions, model.predict(x, verbose=0).reshape(1,-1)[0]])
        labels = np.concatenate([labels, np.array(y[0])])

    predictions = [0 if x < 0.5 else 1 for x in predictions]

    accuracy = accuracy_score(labels, predictions)
    print("Test Accuracy:", accuracy)
    
    f1score = f1_score(labels, predictions)
    print("F1 score:", f1score)
    
    ConfusionMatrixDisplay.from_predictions(labels, predictions)
    
    to_file = len(save_path) > 0
    
    if to_file:
        plt.savefig(f'{save_path}/{model.name}-confussion-matrix.png', facecolor='white', transparent=False, dpi=500)
    
    RocCurveDisplay.from_predictions(labels, predictions)
    
    if to_file:
        plt.savefig(f'{save_path}/{model.name}-roc-curve.png', facecolor='white', transparent=False, dpi=500)

# preprocess data for pretrained model with dedicated method
def preprocess_datasets(train_ds, validation_ds, test_ds, preprocess_input):
    ## Preprocessing input
    train_dataset = train_ds.map(lambda images, labels:( preprocess_input(images), labels))
    validation_dataset = validation_ds.map(lambda images, labels: (preprocess_input(images), labels))
    test_dataset = test_ds.map(lambda images, labels: (preprocess_input(images), labels))
    
    return train_dataset, validation_dataset, test_dataset


# run: build, compile, train and test model
def run_pretrained(train_ds, validation_ds, test_ds, input_shape, preprocess_input, base_model, build_model,
        flatten = True, lr = 0.001, epochs=100, patience=15, save_path=''):
    
    train_ds, validation_ds, test_ds = preprocess_datasets(train_ds, validation_ds, test_ds, preprocess_input)
      
    print_parameters(get_batch_size(train_ds), lr, epochs)
    
    opt = Adam(learning_rate=lr)
    
    model = build_model(opt, base_model, flatten)
    
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=validation_ds, 
        callbacks=[early_stop(patience)],
        verbose=1)
    
    plot_accuracy(history, save_path, model.name)
    
    plot_loss(history, save_path, model.name)

    test_model(test_ds, model, save_path)
    
    del opt, model, history, train_ds, validation_ds, test_ds
    gc.collect()
    
# evalute model
def test_model_raw_data(X_test, y_test, model, save_path):
    predictions = model.predict(X_test, verbose=0)

    predictions = [0 if x < 0.5 else 1 for x in predictions]

    accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy:", accuracy)
    
    f1score = f1_score(y_test, predictions)
    print("F1 score:", f1score)
    
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    
    to_file = len(save_path) > 0
    
    if to_file:
        plt.savefig(f'{save_path}/{model.name}-confussion-matrix.png', facecolor='white', transparent=False, dpi=500)
    
    RocCurveDisplay.from_predictions(y_test, predictions)
    
    if to_file:
        plt.savefig(f'{save_path}/{model.name}-roc-curve.png', facecolor='white', transparent=False, dpi=500)
    
def run(X_train, X_test, y_train, y_test, input_shape, build_model, 
        lr = 0.001, epochs=100, patience=15, batch_size=32, val_split=0.2, save_path=''):
    
    print_parameters(batch_size, lr, epochs)
    
    opt = Adam(learning_rate=lr)
    
    model = build_model(input_shape, opt)
    
    plot_model(model, to_file='model_plot{}.png'.format(model._name), show_shapes=True, show_layer_names=True)

    history = model.fit(
        x=X_train, 
        y=y_train.reshape(-1,1),
        batch_size=batch_size,
        validation_split=val_split,
        epochs=epochs, 
        callbacks=[early_stop(patience)],
        verbose=1)
    
    plot_accuracy(history, save_path, model.name)
    
    plot_loss(history, save_path, model.name)

    test_model_raw_data(X_test, y_test, model, save_path)
    
    del opt, model, history, X_train, y_train, X_test, y_test
    gc.collect()
    