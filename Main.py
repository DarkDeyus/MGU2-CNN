import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import cifar10, cifar100
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from enum import Enum
import itertools
from keras import backend as K


class TestSet(Enum):
    CIFAR10 = 1
    CIFAR100 = 2


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

INIT_LR = 1e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 2
VALIDATION_SPLIT = 0.2
USE_GENERATOR = True
SET = TestSet.CIFAR100


def get_num_classes(test_set):
    if test_set is TestSet.CIFAR10:
        return 10
    if test_set is TestSet.CIFAR100:
        return 100


def get_dataset(test_set):
    if test_set is TestSet.CIFAR10:
        return cifar10.load_data()
    if test_set is TestSet.CIFAR100:
        return cifar100.load_data()


def get_labels(test_set):
    if test_set is TestSet.CIFAR10:
        return CIFAR10_CLASSES
    if test_set is TestSet.CIFAR100:
        return CIFAR100_CLASSES


def normalise(x):
    return (x / 255) - 0.5


def load_dataset(test_set):
    (x_train_loaded, y_train_loaded), (x_test_loaded, y_test_loaded) = get_dataset(test_set)
    num_classes = get_num_classes(test_set)
    # normalisation
    x_train_norm = normalise(x_train_loaded)
    x_test = normalise(x_test_loaded)
    # make validation set
    x_train, x_validation, y_train_to_categorical, y_validation_to_categorical = train_test_split(x_train_norm,
                                                                                                  y_train_loaded,
                                                                                                  test_size=VALIDATION_SPLIT,
                                                                                                  stratify=y_train_loaded)
    # One hot encode
    y_train = keras.utils.to_categorical(y_train_to_categorical, num_classes)
    y_test = keras.utils.to_categorical(y_test_loaded, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_to_categorical, num_classes)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def load_dataset_generator(test_set):
    (x_train, y_train), (x_test, y_test) = get_dataset(test_set)
    num_classes = get_num_classes(test_set)

    x_train = normalise(x_train)
    x_test = normalise(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=VALIDATION_SPLIT)

    datagen.fit(x_train)

    train_generator = datagen.flow(x_train, y_train, subset='training', batch_size=BATCH_SIZE)
    valid_generator = datagen.flow(x_train, y_train, subset='validation', batch_size=BATCH_SIZE)

    return x_train, y_train, x_test, y_test, train_generator, valid_generator


def make_model(test_set):
    num_classes = get_num_classes(test_set)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, input_shape=(256,)))
    model.add(Dense(num_classes))
    model.add(LeakyReLU(0.1))
    model.add(Activation('softmax'))
    return model


def make_model_3_block(test_set):
    num_classes = get_num_classes(test_set)

    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
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

    model.add(Dense(num_classes, activation='softmax'))
    return model


def set_memory_usage():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def main():
    set_memory_usage()

    model = make_model_3_block(SET)
    model.compile(loss='categorical_crossentropy',  # we train 10-way classification
                  optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
                  metrics=['accuracy'])  # report accuracy during training

    if USE_GENERATOR:
        x_train, y_train, x_test, y_test, train_gen, valid_gen = load_dataset_generator(SET)
        history = model.fit_generator(generator=train_gen, validation_data=valid_gen,
                                      steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
                                      epochs=EPOCHS, shuffle=True, verbose=2)

    else:
        x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset(SET)

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

    show_confusion_matrix(prediction, np.argmax(y_test, axis=1), get_labels(SET), SET)
    summarize_diagnostics(history)


def show_confusion_matrix(y_pred, y_real, target_names,test_set):
    cm = confusion_matrix(y_pred, y_real)
    cmap = plt.get_cmap('Blues')
    if test_set is TestSet.CIFAR10:
        plt.figure(figsize=(8, 6))
    else:
        plt.figure(figsize=(40, 30))
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
    plt.ylabel('Poprawna klasa')
    plt.xlabel('Przewidziana klasa')
    plt.show()


def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    # Cross Entropy Loss
    plt.title('Wartość entropii krzyżowej')
    plt.plot(history.history['loss'], color='blue', label='Zbiór treningowy')
    plt.plot(history.history['val_loss'], color='orange', label='Zbiór walidacyjny')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Dokładność klasyfikacji')
    plt.plot(history.history['accuracy'], color='blue', label='Zbiór treningowy')
    plt.plot(history.history['val_accuracy'], color='orange', label='Zbiór walidacyjny')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
