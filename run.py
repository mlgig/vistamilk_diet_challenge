from pathlib import Path
from datetime import datetime

import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from rich.console import Console

from src import utils


random_state = 0
epochs = 10_000
patience = 500

console = Console()

full_data = Path('data', 'full_dataset.xlsx')
train_path = Path('data', 'raw_train.csv')
test_path = Path('data', 'raw_test.csv')

train_df = utils.read_raw_data(full_data, train_path, 0)
test_df = utils.read_raw_data(full_data, test_path, 1)

# data preparation: outlier removal and shuffling
train_df = train_df[train_df['col1'] >= 1]
train_df = train_df.sample(frac=1, random_state=random_state)

X_pandas = train_df.drop(['Diet'], axis=1)
y = train_df['Diet']

# train_waves = utils.convert_waves(X_pandas, normalize=True)
train_waves = utils.convert_waves_linear(X_pandas, normalize=True)

y = y.map({'GRS': 0, 'CLV': 1, 'TMR': 2}).values
train_labels = y.reshape((y.shape[0], 1))

y = tf.one_hot(y, depth=3).numpy()

T_X, test_X, T_y, test_y = train_test_split(
    train_waves, y,
    stratify=train_labels, test_size=0.2, random_state=random_state)

training_X, validation_X, training_y, validation_y = train_test_split(
    T_X, T_y, stratify=T_y, test_size=0.25)

console.print(f'Training set shape: {training_X.shape}')
console.print(f'Validation set shape: {validation_X.shape}')
console.print(f'Test set shape: {test_X.shape}')

log_dir = Path('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=patience)

model = tf.keras.Sequential([
    layers.Reshape((1060, 1), input_shape=(1060, 1, 1)),
    # layers.InputLayer(input_shape=(1060, 1, 1)),
    layers.Conv1D(128, 5),
    layers.Conv1D(64, 3),
    layers.Conv1D(32, 3),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(training_X, training_y,
                    validation_data=(validation_X, validation_y),
                    epochs=epochs,
                    batch_size=1946,
                    callbacks=[earlystopping_callback, tensorboard_callback])


test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
console.print(f'Test accuracy: {test_acc}')
