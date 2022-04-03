import sys
import importlib

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sktime.classification.dictionary_based import IndividualBOSS, BOSSEnsemble
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier,ProximityForest
from sktime.utils.data_processing import from_2d_array_to_nested


def convert_to_ts(df):
    return from_2d_array_to_nested(df)


def temporal_performance(X_train, y):

    X_train = convert_to_ts(X_train)


    classifiers = {
#         'Boss': IndividualBOSS(),
#         'BossEnsemble': BOSSEnsemble(max_ensemble_size=5),
#         'dtw': KNeighborsTimeSeriesClassifier(),
        'PF': ProximityForest(),
#         'SVM': SVC(),
#         'adaboost': AdaBoostClassifier(),
#         'mlp': MLPClassifier()
    }

    for cls_name, classifier in classifiers.items():
        score, pred, report, confusion = utils.basics(classifier, X_train, y)

    #     print(f'[bold underline magenta]{cls_name}[/bold underline magenta]')
        print(f'{cls_name} Scores: {score} (avg. {score.mean()})')
        print(report)
        print(confusion)
        print("--"*20)


full_data = Path('..', 'data', 'full_dataset.xlsx')
train_path = Path('..', 'data', 'raw_train.csv')
test_path = Path('..', 'data', 'raw_test.csv')

train_df = utils.read_raw_data(full_data, train_path, 0)
test_df = utils.read_raw_data(full_data, test_path, 1)


train_df.drop(train_df[train_df.col1<1].index.tolist(), axis=0, inplace=True)
test_df.drop(test_df[test_df.col1<1].index.tolist(), axis=0, inplace=True)


train_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)

train_df = train_df.sample(frac=1)

X_train = train_df.drop(['Diet'], axis=1)
y = train_df['Diet']

def tab_performance(X_train, y):
    classifiers = {
#         'knn': KNeighborsClassifier(),
        'ridge': RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
#         'tree': DecisionTreeClassifier(),
        'forest': RandomForestClassifier(),
#         'SVM': SVC(),
        'adaboost': AdaBoostClassifier(),
#         'mlp': MLPClassifier()
    }

    for cls_name, classifier in classifiers.items():
        score, pred, report, confusion = utils.basics(classifier, X_train, y)

    #     print(f'[bold underline magenta]{cls_name}[/bold underline magenta]')
        print(f'{cls_name} Scores: {score} (avg. {score.mean()})')
        print(report)
        print(confusion)
        print("--"*20)


temporal_performance(X_train, y)
