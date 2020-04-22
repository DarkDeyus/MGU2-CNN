import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras import backend as K

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
INIT_LR = 1e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10
TEST_SIZE = 0.1

def load_dataset():
    (x_train_loaded, y_train_loaded), (x_test_loaded, y_test_loaded) = cifar10.load_data()
    # normalisation
    x_train_norm = (x_train_loaded / 255) - 0.5
    x_test = (x_test_loaded / 255) - 0.5
    # make validation set
    x_train, x_validation, y_train_to_categorical, y_validation_to_categorical = train_test_split(x_train_norm, y_train_loaded, test_size=TEST_SIZE)
    # One hot encode
    y_train = keras.utils.to_categorical(y_train_to_categorical, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test_loaded, NUM_CLASSES)
    y_validation = keras.utils.to_categorical(y_validation_to_categorical, NUM_CLASSES)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


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

def make_model_3_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))


def set_memory_usage():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def main():
    set_memory_usage()
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset()
    model = make_model()
    model.compile(loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy'])  # report accuracy during training

    history = model.fit(
        x_train, y_train,  # prepared data
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_validation, y_validation),  # validation_set
        shuffle=True,
        verbose=2,
        initial_epoch=0)

    _, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("Accuracy = %.2f " % (acc * 100.0) + "%")
    prediction = model.predict_classes(x_test, batch_size=BATCH_SIZE)
    show_confusion_matrix(prediction, np.argmax(y_test, axis=1), cifar10_classes)
    summarize_diagnostics(history)


def show_confusion_matrix(y_pred, y_real, target_names):
    cm = confusion_matrix(y_pred, y_real)
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Macierz pomyłek")
    plt.colorbar()
    # set labels
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange',  label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
