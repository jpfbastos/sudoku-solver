import numpy as np
import pandas as pd
import os
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

model = Sequential()


def start(dataset):
    global model
    num_classes = 0
    if os.path.exists('model_' + dataset + '.keras'):
        # Checks model exists
        model = load_model('model_' + dataset + '.keras')
        print('Loaded saved model from disk.')
    else:
        if dataset == 'mnist':
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            # Reshape it to (samples, rows, cols, channels) to be in the correct format for 2d convolution
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
            # Normalisation
            X_train = X_train / 255
            X_test = X_test / 255
            # Converts the vector of labels into a binary matrix
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            num_classes = y_test.shape[1]
        elif dataset == 'tmnist':  # Typographical mnist (uses fonts instead of handwritten digits)
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
                Input(shape=(28, 28, 1)),
                Conv2D(32, (5, 5), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(16, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.3),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')
            ]
        )
        # Allows stopping training when improvements to the validation data are small to avoid overfitting
        callback = EarlyStopping(monitor='val_loss', patience=2)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=300, epochs=50,
                  callbacks=[callback], shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print('CNN Error: %.2f%%' % (100 - scores[1] * 100))

        # Serialise model to .keras
        model.save('model_' + dataset + '.keras')


def extract_number(digit):
    global model
    prediction = model.predict(digit)
    predicted_class = np.argmax(prediction, axis=1)
    print(predicted_class[0])
    return predicted_class[0]
