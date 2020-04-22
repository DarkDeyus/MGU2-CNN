import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, Activation
from keras import backend as K

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalisation
    x_train2 = (x_train / 255) - 0.5
    x_test2 = (x_test / 255) - 0.5
    # One hot encode
    y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train2, y_train2, x_test2, y_test2


def make_model():
    model = Sequential()
    # to check
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, input_shape=(256,)))
    model.add(Dense(NUM_CLASSES))
    model.add(LeakyReLU(0.1))
    model.add(Activation('softmax'))
    return model


def main():
    x_train, y_train, x_test, y_test = load_dataset()
    model = make_model()
    model.compile(loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy'])  # report accuracy during training

    model.fit(
        x_train, y_train,  # prepared data
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),  # get validation set maybe
        shuffle=True,
        verbose=1,
        initial_epoch=0)


if __name__ == "__main__":
    main()
