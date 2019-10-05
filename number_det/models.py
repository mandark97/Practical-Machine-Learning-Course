from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     InputLayer, MaxPool2D)

from tensorflow.keras import Sequential


def lenet():
    model = Sequential([
        InputLayer((28, 84, 1)),
        Conv2D(6, kernel_size=5, strides=1, activation='relu'),
        MaxPool2D(2, strides=2),
        Conv2D(16, kernel_size=5, strides=1, activation='relu'),
        MaxPool2D(2, strides=2),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(256, activation='softmax'),
    ])
    return model


def changed_lenet():
    model = Sequential([
        Conv2D(6, kernel_size=3,
               activation='relu', input_shape=(28, 84, 1)),
        Conv2D(6, kernel_size=3, activation='relu'),
        Conv2D(6, kernel_size=3, activation='relu'),
        MaxPool2D(2, strides=2),
        Conv2D(16, kernel_size=3, activation='relu'),
        Conv2D(16, kernel_size=3, activation='relu'),
        Conv2D(16, kernel_size=3, activation='relu'),
        MaxPool2D(2, strides=2),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(256, activation='softmax')])
    return model


def vgg():
    model = Sequential([
        Conv2D(6, (3, 3), activation='relu',
               padding='same', input_shape=(28, 84, 1)),
        Conv2D(6, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='softmax')
    ])

    return model


def vgg_2():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               padding='same', input_shape=(28, 84, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPool2D((2, 2), strides=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='softmax')
    ])
    return model
