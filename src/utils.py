import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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


def convert_to_image(wave):
    '''Convert wave to image-like array
    
    To make the wave image-like, the dimensions should be 33x33.
    As 1060 is the original size, closest square is 1089 = 33^2.
    For RGB conversion, the resulting image would only be 19x19
    with a padding of 23.

    The normalization step could also go within the network.
    '''
    # wave *= 255 / wave.max()         # normalize
    wave = np.pad(wave, (0, 29))     # pad
    wave = wave.reshape((33, 33, 1)) # reshape

    return wave


def convert_waves(df):
    '''Apply conversion function to the entire dataset.
    
    This should probably be optimized.
    '''
    waves = []

    for _, row in df.iterrows():
        conv = convert_to_image(row)
        waves.append(conv)

    waves = np.vstack(waves)
    return waves.reshape((df.shape[0], 33, 33, 1))