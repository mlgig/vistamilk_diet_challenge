import argparse
import json

from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from rich.console import Console

from src import utils
from src import vista_models


parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', type=str)
parser.add_argument('--arch', '-a', choices=['fcn', 'cnn', 'cnn_dilated'])
parser.add_argument('--data', '-d', choices=['full', 'no_water'])

args = parser.parse_args()
console = Console()

run_name = args.name
arch = args.arch
data = args.data

epochs = 12_000

log_dir = Path('logs', 'fit', run_name)

full_data = Path('data', 'full_dataset.xlsx')
train_path = Path('data', 'raw_train.csv')
test_path = Path('data', 'raw_test.csv')

train_df = utils.read_raw_data(full_data, train_path, 0)
test_df = utils.read_raw_data(full_data, test_path, 1)

train_df = train_df[train_df['col1'] >= 1]
test_df = test_df[test_df['col1'] >= 1]

train_df = train_df.sample(frac=1, random_state=0)

X_pandas = train_df.drop(['Diet'], axis=1)
y = train_df['Diet']

cut_water = data == 'no_water'

if arch == 'fcn':
    train_waves = utils.convert_waves_linear(
        X_pandas, normalize=True, cut_water=cut_water)
    test_waves = utils.convert_waves_linear(
        test_df, normalize=True, cut_water=cut_water)
else:
    train_waves = utils.convert_waves(
        X_pandas, normalize=True, cut_water=cut_water)
    test_waves = utils.convert_waves(
        test_df, normalize=True, cut_water=cut_water)

y = y.map({'GRS': 0, 'CLV': 1, 'TMR': 2}).values
train_labels = y.reshape((y.shape[0], 1))

y = tf.one_hot(y, depth=3).numpy()

model, _ = utils.get_model(arch, data, log_dir)

model.summary()

history = model.fit(train_waves, y,
                    epochs=epochs,
                    batch_size=train_waves.shape[0])

model.save(Path('models', run_name))
json.dump(history.history,
          open(Path('models', run_name, run_name + '.json'), 'w'))

# keep this once final architecture has been selected
predictions = model.predict(test_waves)
logits = [np.argmax(x) for x in predictions]

c = {0: 'GRS', 1: 'CLV', 2: 'TMR'}
classes = [*map(c.get, logits)]

console.log('Predictions done.')

grs = classes.count('GRS')
clv = classes.count('CLV')
tmr = classes.count('TMR')

test_dataset = pd.read_csv('data/raw_test.csv')
test_dataset = test_dataset[test_dataset['col1'] >= 1]

test_dataset['Diet'] = classes

test_dataset.to_csv('data/raw_test_predictions.csv', index=None)

console.print(f'GRS: {grs}, CLV: {clv}, TMR: {tmr}')
