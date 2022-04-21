import tensorflow.keras as K

from tensorflow.keras import layers


GLB_LR             = 0.00001
GLB_PATIENTE       = 1000
GLB_DROPOUT        = 0.2
FCN_SHAPE_FULL     = (1060, 1, 1)
FCN_SHAPE_NO_WATER = (518, 1, 1)
CNN_SHAPE_FULL     = (33, 33, 1)
CNN_SHAPE_NO_WATER = (23, 23, 1)
HIDDEN_ACTIVATION  = 'elu'
OUT_ACTIVATION     = 'softmax'



def get_callbacks(log_dir):
    return {
        'tensorboard': K.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1),
        'early_stopping': K.callbacks.EarlyStopping(
            monitor='val_loss', patience=GLB_PATIENTE,
            restore_best_weights=True)
    }


def fcn(input_shape, log_dir):
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


def cnn(input_shape, log_dir):
    model = K.Sequential([
        layers.Conv2D(32, (3, 3), activation=HIDDEN_ACTIVATION, input_shape=input_shape),
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


def cnn_dilated(input_shape, log_dir):
    model = K.Sequential([
        layers.Conv2D(32, (3, 3), dilation_rate=(2, 2), activation=HIDDEN_ACTIVATION, input_shape=input_shape),
        layers.Conv2D(64, (2, 2), dilation_rate=(2, 2), activation=HIDDEN_ACTIVATION),
        layers.Conv2D(128, (2, 2), dilation_rate=(2, 2), activation=HIDDEN_ACTIVATION),
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
