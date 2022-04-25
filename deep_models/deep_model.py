import argparse
import json

from pathlib import Path

import tensorflow as tf
import numpy as np

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

epochs = 50_000

log_dir = Path('logs', 'fit', run_name)

full_data = Path('data', 'full_dataset.xlsx')
train_path = Path('data', 'raw_train.csv')
test_path = Path('data', 'raw_test.csv')

train_df = utils.read_raw_data(full_data, train_path, 0)
test_df = utils.read_raw_data(full_data, test_path, 1)

train_df = train_df[train_df['col1'] >= 1]
train_df = train_df.sample(frac=1, random_state=0)

X_pandas = train_df.drop(['Diet'], axis=1)
y = train_df['Diet']

cut_water = data == 'no_water'

if arch == 'fcn':
    train_waves = utils.convert_waves_linear(
        X_pandas, normalize=True, cut_water=cut_water)
else:
    train_waves = utils.convert_waves(
        X_pandas, normalize=True, cut_water=cut_water)

y = y.map({'GRS': 0, 'CLV': 1, 'TMR': 2}).values
train_labels = y.reshape((y.shape[0], 1))

y = tf.one_hot(y, depth=3).numpy()

T_X, test_X, T_y, test_y = train_test_split(
    train_waves, y, stratify=train_labels, test_size=0.4, random_state=0)

training_X, validation_X, training_y, validation_y = train_test_split(
    T_X, T_y, stratify=T_y, test_size=0.2)

model, callbacks = utils.get_model(arch, data, log_dir)

tensorboard_callback = callbacks['tensorboard']
earlystopping_callback = callbacks['early_stopping']

model.summary()

history = model.fit(training_X, training_y,
                    validation_data=(validation_X, validation_y),
                    epochs=epochs,
                    batch_size=training_X.shape[0],
                    callbacks=[earlystopping_callback, tensorboard_callback])

model.save(Path('models', run_name))
json.dump(history.history,
          open(Path('models', run_name, run_name + '.json'), 'w'))

# keep this once final architecture has been selected
test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
console.print(f'Test accuracy: {test_acc} (loss {test_loss})')

with open(Path('models', run_name, 'test_results.txt'), 'w') as f:
    f.write(f'accuracy: {test_acc}\nloss: {test_loss}\n')
