import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.src.saving.legacy.model_config import model_from_json
from keras.utils import to_categorical
import pandas as pd
import os

from sklearn.model_selection import train_test_split

model = Sequential()


def start(dataset='tmnist'):
    global model
    num_classes = 0
    if os.path.exists('model_' + dataset + '.json') and os.path.exists('model_' + dataset + '.h5'):
        json_file = open('model_' + dataset + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('model_' + dataset + '.h5')
        print("Loaded saved model from disk.")
    else:
        if dataset == 'mnist':
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
            X_train = X_train / 255
            X_test = X_test / 255
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            num_classes = y_test.shape[1]
        elif dataset == 'tmnist':
            df = pd.read_csv('TMNIST_Data.csv').to_numpy()
            np.random.shuffle(df)
            data = df[:, 2:] / 255
            labels = df[:, 1]
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            num_classes = y_test.shape[1]
        else:
            print('Invalid Dataset. Options are \'mnist\' or \'tmnist\'')

        model = Sequential(
            [
                Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(16, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.3),
                Flatten(),
                #Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')
            ]
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=300, epochs=50,
                  callbacks=[callback], shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

        # serialize model to JSON
        model_json = model.to_json()
        with open('model_' + dataset + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('model_' + dataset + '.h5')
        print("Saved model to disk")


def extract_number(digit):
    global model
    prediction = model.predict(digit)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class[0]
