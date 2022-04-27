from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow.keras as K
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from src.vista_models import fcn, cnn, cnn_dilated
from src.vista_models import FCN_SHAPE_FULL, FCN_SHAPE_NO_WATER
from src.vista_models import CNN_SHAPE_FULL, CNN_SHAPE_NO_WATER
from src.vista_models import CNN_PAD_FULL, CNN_PAD_NO_WATER
from src.vista_models import ModelObj


def read_raw_data(full_path: Path, dest_path: Path, sheet: int) -> pd.DataFrame:
    '''Read raw data from filesystem

    The original data are provided into an Excel file. This method reads the
    file, and then saves the resulting csv into a new file. If the csv file
    already exists, the Excel file won't be considered (thus saving us precious
    seconds!).

    Arguments
    ---------
    full_path: Path
        The path of the original Excel file to read.
    dest_path: Path
        The destination path of the produced csv file.
    sheet: int
        The sheet number to read in the Excel file.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the required data.
    '''
    df = None

    if dest_path.exists():
        df = pd.read_csv(dest_path)
    else:
        df = pd.read_excel(full_path, sheet, engine='openpyxl')
        df.to_csv(dest_path, index=False)
        
    return df


def convert_to_image(wave: list, cut_water: bool = False) -> np.array:
    '''Convert wave to image-like array

    Read in a wave, and convert it to matrix. Shape and padding depend on
    whether water regions are excluded:

    - Included water regions: shape (33, 33, 1) with padding 29
    - Excluded water regions: shape (23, 23, 1) with padding 11

    Arguments
    ---------
    wave: list
        The original wave to reshape.
    cut_water: bool
        If true, water regions are not supposed to be included in the wave.

    Returns
    -------
    np.array
        The appropriately reshaped wave, including padding of zeros.
    '''
    shape = CNN_SHAPE_NO_WATER if cut_water else CNN_SHAPE_FULL
    pad = CNN_PAD_NO_WATER if cut_water else CNN_PAD_FULL

    wave = np.pad(wave, (0, pad))  # pad
    wave = wave.reshape(shape)     # reshape

    return wave


def convert_waves(df: pd.DataFrame,
                  normalize: bool = False,
                  cut_water: bool = False) -> np.ndarray:
    '''Convert waves into image-like structure
    
    This method converts the provided dataset into a ndarray of squared,
    bidimensional images. The shape of the resulting image depends on whether
    the water regions are included or excluded.

    Arguments
    ---------
    df: pd.DataFrame
        The original dataframe containing the waves.
    normalize: bool
        When true, waves will be normalized.
    cut_water: bool
        When true, water regions will be excluded.

    Returns
    -------
    np.ndarray
        A vertically stacked array containing all the waves.
    '''
    waves = []
    df = prep_df(df, normalize, cut_water)

    for _, row in df.iterrows():
        conv = convert_to_image(row, cut_water=cut_water)
        waves.append(conv)

    waves = np.vstack(waves)

    btc = df.shape[0]
    shape = (btc, *CNN_SHAPE_NO_WATER) if cut_water else (btc, *CNN_SHAPE_FULL)
    waves = waves.reshape(shape)

    return waves


def convert_waves_linear(df: pd.DataFrame,
                         normalize: bool = False,
                         cut_water: bool = False) -> np.ndarray:
    '''Read in waves linearly

    Instead of loading waves as images, read them as individual sequences and
    stack them. Decide whether on not to normalise them, and whether or not
    water regions should be retained.

    Arguments
    ---------
    df: pd.DataFrame
        The original dataframe containing the waves.
    normalize: bool
        When true, waves will be normalized.
    cut_water: bool
        When true, water regions will be excluded.

    Returns
    -------
    np.ndarray
        A vertically stacked array containing all the waves.
    '''
    df = prep_df(df, normalize, cut_water)
    waves = [wave.to_numpy() for _, wave in df.iterrows()]

    return np.vstack(waves)


def prep_df(df: pd.DataFrame, normalize: bool, cut_water: bool) -> pd.DataFrame:
    '''Prepare pandas frame for conversion

    This method receives a pandas dataframe and applies 2 operations, namely,
    normalization and chopping of water regions. If both transformations are
    required, chopping will be performed before normalization.

    Arguments
    ---------
    df: pd.DataFrame
        The original dataframe to prepare for conversion.
    normalize: bool
        When true, standard normalization will be applied to the dataframe.
    cut_water: bool
        When true, water regions will be excluded from the dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe where normalization and chopping is applied.

    '''
    if cut_water:
        df = df.iloc[:, np.r_[0:171, 206:535, 729:747]]

    if normalize:
        df = (df - df.mean()) / df.std()
    
    return df


def get_model(arch: str, data: str, log_dir: Path) -> ModelObj:
    '''Create learning model

    This method is a small wrapper to quickly create a model to train. The
    architecture, data modality and log directory are required so that the model
    structure can conform and the logging callback can be properly set.

    Arguments
    ---------
    arch: str
        The model architecture. It can be fcn for the fully connected model, cnn
        for the convolutional model with regular filters, and cnn_dilated for
        the convolutional model with dilated filters.
    data: str
        Data modality. It can be full for models that accept full waves (so
        including water regions) or no_water for models that accept waves with
        no water regions.
    log_dir: Path
        The path where the training history should be saved (this is done
        automatically through the tensorboard callback).

    Returns
    -------
    ModelObj
        A tuple that contains the model as first object, and a dictionary of
        callbacks as second object. Included callbacks are "tensorboard" and
        "early_stopping".
    '''
    models = {
        'fcn': {
            'full': lambda: fcn(FCN_SHAPE_FULL, log_dir),
            'no_water': lambda: fcn(FCN_SHAPE_NO_WATER, log_dir),
        },
        'cnn': {
            'full': lambda: cnn(CNN_SHAPE_FULL, log_dir),
            'no_water': lambda: cnn(CNN_SHAPE_NO_WATER, log_dir)
        },
        'cnn_dilated': {
            'full': lambda: cnn_dilated(CNN_SHAPE_FULL, log_dir),
            'no_water': lambda: cnn_dilated(CNN_SHAPE_NO_WATER, log_dir)
        }
    }

    return models[arch][data]()


def basics(clf, X, y, cv=5):
    validator = StratifiedKFold(n_splits=cv)

    scores = cross_val_score(clf, X, y, cv=validator)
    predictions = cross_val_predict(clf, X, y, cv=cv)
    confusion = confusion_matrix(predictions, y)
    report = classification_report(predictions, y)

    return scores, predictions, report, confusion
