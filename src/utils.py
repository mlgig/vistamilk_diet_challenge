import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from src.vista_models import fcn, cnn, cnn_dilated
from src.vista_models import FCN_SHAPE_FULL, FCN_SHAPE_NO_WATER
from src.vista_models import CNN_SHAPE_FULL, CNN_SHAPE_NO_WATER


def read_raw_data(full_path, dest_path, sheet):
    if dest_path.exists():
        return pd.read_csv(dest_path)
    else:
        df = pd.read_excel(full_path, sheet, engine='openpyxl')
        df.to_csv(dest_path, index=False)
        
        return df


def plot_wave(df, target_col=None):
    fig = plt.figure(figsize=(25, 10))

    label = df[target_col] if target_col else 'no target'

    plt.plot(df.drop([target_col], axis=0), label=label)
    fig.axes[0].xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.legend(prop={'size': 25})
    plt.grid()

    return fig


def basics(clf, X, y, cv=5):
    validator = StratifiedKFold(n_splits=cv)
    scores = cross_val_score(clf, X, y, cv=validator)
    predictions = cross_val_predict(clf, X, y, cv=cv)
    confusion = confusion_matrix(predictions, y)
    report = classification_report(predictions, y)

    return scores, predictions, report, confusion


def convert_to_image(wave, cut_water=False):
    '''Convert wave to image-like array
    
    To make the wave image-like, the dimensions should be 33x33.
    As 1060 is the original size, closest square is 1089 = 33^2.
    For RGB conversion, the resulting image would only be 19x19
    with a padding of 23.

    The normalization step could also go within the network.
    '''
    shape = (23, 23, 1) if cut_water else (33, 33, 1)
    pad = 11 if cut_water else 29

    wave = np.pad(wave, (0, pad))  # pad
    wave = wave.reshape(shape)     # reshape

    return wave


def convert_waves(df, normalize=False, cut_water=False):
    '''Apply conversion function to the entire dataset.
    
    This should probably be optimized.
    '''
    waves = []

    if cut_water:
        df = df.iloc[:, np.r_[0:171, 206:535, 729:747]]

    if normalize:
        df = (df - df.mean()) / df.std()

    for _, row in df.iterrows():
        conv = convert_to_image(row, cut_water=cut_water)
        waves.append(conv)

    waves = np.vstack(waves)
    shape = (df.shape[0], 23, 23, 1) if cut_water else (df.shape[0], 33, 33, 1)
    waves = waves.reshape(shape)

    return waves


def convert_waves_linear(df, normalize=False, cut_water=False):
    '''Read waves linearly

    Instead of loading waves as images, read them as individual sequences and
    stack them.
    '''

    if cut_water:
        df = df.iloc[:, np.r_[0:171, 206:535, 729:747]]

    if normalize:
        df = (df - df.mean()) / df.std()

    waves = [wave.to_numpy() for _, wave in df.iterrows()]

    return np.vstack(waves)


def get_model(arch, data, log_dir):
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
