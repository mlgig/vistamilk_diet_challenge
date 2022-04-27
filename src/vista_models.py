from pathlib import Path
from typing import Tuple

import tensorflow.keras as K

from tensorflow.keras import layers


ModelObj = Tuple[K.Sequential, dict]

GLB_LR             = 0.00001
GLB_PATIENTE       = 1500
GLB_DROPOUT        = 0.2
FCN_SHAPE_FULL     = (1060, 1, 1)
FCN_SHAPE_NO_WATER = (518, 1, 1)
CNN_SHAPE_FULL     = (33, 33, 1)
CNN_SHAPE_NO_WATER = (23, 23, 1)
CNN_PAD_FULL       = 29
CNN_PAD_NO_WATER   = 11
HIDDEN_ACTIVATION  = 'elu'
OUT_ACTIVATION     = 'softmax'



def get_callbacks(log_dir: Path) -> dict:
    '''Create callbacks and wrape them in a dictionary

    As per title. Arrange callbacks in a dictionary.

    Arguments
    ---------
    log_dir: Path
        The directory to store the training data that can be used by the
        tensorboard callback.

    Returns
    -------
    dict
        A dictionary containing 2 callbacks: tensorboard and early_stopping.
    '''
    return {
        'tensorboard': K.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1),
        'early_stopping': K.callbacks.EarlyStopping(
            monitor='val_loss', patience=GLB_PATIENTE,
            restore_best_weights=True)
    }


def fcn(input_shape: tuple, log_dir: Path) -> ModelObj:
    '''Fully connected model

    Create a fully connected model. Data modality depends on the input shape.

    Arguments
    ---------
    input_shape: tuple
        The input shape of the data.
    log_dir: Path
        The log directory for the tensorboard callback.

    Returns
    -------
    ModelObj
        The compiled model with the callback dictionary.
    '''
    model = K.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(1024, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(512, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(256, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(128, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(64, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(32, activation=HIDDEN_ACTIVATION),
        layers.Dropout(GLB_DROPOUT),
        layers.Dense(3, activation=OUT_ACTIVATION)])

    model.compile(optimizer=K.optimizers.Adam(learning_rate=GLB_LR),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model, get_callbacks(log_dir)


def cnn(input_shape: tuple, log_dir: Path) -> ModelObj:
    '''Convolutional model

    Create a convolutional model. Data modality depends on the input shape.

    Arguments
    ---------
    input_shape: tuple
        The input shape of the data.
    log_dir: Path
        The log directory for the tensorboard callback.

    Returns
    -------
    ModelObj
        The compiled model with the callback dictionary.
    '''
    model = K.Sequential([
        layers.Conv2D(32, (3, 3), activation=HIDDEN_ACTIVATION,
                      input_shape=input_shape),
        layers.Conv2D(64, (2, 2), activation=HIDDEN_ACTIVATION),
        layers.Conv2D(128, (2, 2), activation=HIDDEN_ACTIVATION),
        layers.Flatten(),
        layers.Dense(512, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(256, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(128, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(64, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(32, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(3, activation=OUT_ACTIVATION)])

    model.compile(optimizer=K.optimizers.Adam(learning_rate=GLB_LR),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model, get_callbacks(log_dir)


def cnn_dilated(input_shape: tuple, log_dir: Path) -> ModelObj:
    '''Convolutional dilated model

    Create a convolutional model with dilated filters. Data modality depends on
    the input shape.

    Arguments
    ---------
    input_shape: tuple
        The input shape of the data.
    log_dir: Path
        The log directory for the tensorboard callback.

    Returns
    -------
    ModelObj
        The compiled model with the callback dictionary.
    '''
    model = K.Sequential([
        layers.Conv2D(32, (3, 3), dilation_rate=(2, 2),
                      activation=HIDDEN_ACTIVATION, input_shape=input_shape),
        layers.Conv2D(64, (2, 2), dilation_rate=(2, 2),
                      activation=HIDDEN_ACTIVATION),
        layers.Conv2D(128, (2, 2), dilation_rate=(2, 2),
                      activation=HIDDEN_ACTIVATION),
        layers.Flatten(),
        layers.Dense(512, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(256, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(128, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(64, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(32, activation=HIDDEN_ACTIVATION),
        layers.Dropout(0.2),
        layers.Dense(3, activation=OUT_ACTIVATION)])

    model.compile(optimizer=K.optimizers.Adam(learning_rate=GLB_LR),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model, get_callbacks(log_dir)
