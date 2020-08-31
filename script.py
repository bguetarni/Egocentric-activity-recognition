import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import pickle
from utils import *


def create_model(optimizer):
    inputs = keras.Input(shape=(k, 256, 456, 3))
    conv1 = layers.Conv2D(filters=4, kernel_size=3, strides=2, kernel_regularizer='l2')
    conv2 = layers.Conv2D(filters=8, kernel_size=3, strides=2, kernel_regularizer='l2')
    conv3 = layers.Conv2D(filters=8, kernel_size=3, strides=2, kernel_regularizer='l2')
    conv4 = layers.Conv2D(filters=16, kernel_size=3, strides=2, kernel_regularizer='l2')
    lstm = layers.LSTM(units=100, kernel_regularizer='l2')
    verb_classification = layers.Dense(125, activation='softmax', kernel_regularizer='l2')
    noun_classification = layers.Dense(352, activation='softmax', kernel_regularizer='l2')

    x = layers.TimeDistributed(conv1)(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(conv2)(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(conv3)(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(conv4)(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = lstm(x)
    outputs = verb_classification(x), noun_classification(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


# to use the download script
assert sys.version_info.major == 3 and sys.version_info.minor >= 5, 'Python 3.5+ needed. Please use a more recent Python version'

# Models and Optimizer
# lr = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.1, decay_steps=20, decay_rate=0.1)
lr = 0.1
model = create_model(keras.optimizers.SGD(learning_rate=lr))
# model = create_model(keras.optimizers.Adam(learning_rate=lr))

# metrics
values = []

# code already done
if False:
    # download groun truth
    with requests.Session() as s:
        download = s.get(LABELS_URL)
        content = download.content.decode('utf-8')
        with open(LABELS_PATH, 'w') as f:
            for line in content:
                f.write(line)

    # download the participant RGB frames
    download_video_set([i for i in range(nb_participant)])

# loop over the videos
for i, video in enumerate(os.listdir(DATA_PATH)):
    print('Video: {}'.format(video))

    # create training/validation data
    print('		Creating the training set')
    X, Y = create_training_set(video)

    # train the model
    print('		Training')
    h = model.fit(X, Y, batch_size=32, epochs=1, verbose=1, validation_split=0.25, shuffle=True)

    # save metrics
    values.append(h.history)

    # aplpy learning decay and save weights
    if (i+1) % 50 == 0:
        K.set_value(model.optimizer.learning_rate, 0.1*model.optimizer.learning_rate)
        model.save(DATA_PATH + 'models/model_{}.h5'.format((i+1) % 50))

model.save(DATA_PATH + 'models/SGD_final_model.h5')
with open(DATA_PATH + 'values.pkl', 'wb') as f:
    pickle.dump(values, f)
