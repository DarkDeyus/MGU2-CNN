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
import Models
import Optimizers

class TestSet(Enum):
    CIFAR10 = 1
    CIFAR100 = 2


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
        return ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
    elif test_set is TestSet.CIFAR100:
        return [
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
            'worm']


def normalise(x):
    return (x / 255) - 0.5

def load_dataset(test_set, test_size):
    (x_train_loaded, y_train_loaded), (x_test_loaded, y_test_loaded) = get_dataset(test_set)
    num_classes = get_num_classes(test_set)

    # normalisation
    x_train_norm = normalise(x_train_loaded)
    x_test = normalise(x_test_loaded)

    # make validation set
    x_train, x_validation, y_train_to_categorical, y_validation_to_categorical = train_test_split(x_train_norm,
                                                                                                  y_train_loaded,
                                                                                                  test_size=test_size,
                                                                                                  stratify=y_train_loaded)
    # One hot encode
    y_train = keras.utils.to_categorical(y_train_to_categorical, num_classes)
    y_test = keras.utils.to_categorical(y_test_loaded, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_to_categorical, num_classes)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def load_dataset_generator(test_set, test_size, batch_size):
    (x_train, y_train), (x_test, y_test) = get_dataset(test_set)
    num_classes = get_num_classes(test_set)

    # normalisation
    x_train = normalise(x_train)
    x_test = normalise(x_test)

    # One hot encode
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
        validation_split=test_size)
    datagen.fit(x_train)

    train_generator = datagen.flow(x_train, y_train, subset='training', batch_size=batch_size)
    valid_generator = datagen.flow(x_train, y_train, subset='validation', batch_size=batch_size)

    return x_train, y_train, x_test, y_test, train_generator, valid_generator

def set_memory_usage():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


# def main():
#     set_memory_usage()

#     model = make_model_3_block(CHOSEN_SET)
#     model.compile(loss='categorical_crossentropy',  # we train 10-way classification
#                   optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
#                   metrics=['accuracy'])  # report accuracy during training

#     if USE_GENERATOR:
#         x_train, y_train, x_test, y_test, train_gen, valid_gen = load_dataset_generator(CHOSEN_SET)
#         history = model.fit_generator(generator=train_gen, validation_data=valid_gen,
#                                       steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
#                                       epochs=EPOCHS, shuffle=True, verbose=2)

#     else:
#         x_train, y_train, x_validation, y_validation, x_test, y_test = load_dataset(CHOSEN_SET)
#         history = model.fit(
#             x_train, y_train,  # prepared data
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             validation_data=(x_validation, y_validation),  # validation_set
#             shuffle=True,
#             verbose=2,
#             initial_epoch=0)

#     return history


def display_results(model, batch_size, history, dataset, use_generator, test_set):
    if use_generator:
        _, _, x_test, y_test, _, _ = dataset
    else:
        _, _, _, _, x_test, y_test = dataset
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Accuracy = %.2f " % (acc * 100.0) + "%")
    prediction = model.predict_classes(x_test, batch_size=batch_size)

    # Show plots
    show_confusion_matrix(prediction, np.argmax(y_test, axis=1), test_set)
    show_plots(history)


def show_confusion_matrix(y_pred, y_real, test_set):
    target_names = get_labels(test_set)
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


def show_plots(history):
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

def calculate_for(batch_size, epochs, optimizer, model, dataset, use_generator):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    if use_generator:
        x_train, y_train, x_test, y_test, train_gen, valid_gen = dataset
        history = model.fit_generator(generator=train_gen, validation_data=valid_gen,
                                      steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
                                      epochs=epochs, shuffle=True, verbose=2)
    else:
        x_train, y_train, x_validation, y_validation, x_test, y_test = dataset
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_validation, y_validation),
                            shuffle=True,
                            verbose=2,
                            initial_epoch=0)
    return history

def main():
    #set_memory_usage()
    init_lr = 1e-3
    batch_size = 32
    epochs = 2
    test_size = 0.2
    test_set = TestSet.CIFAR10
    use_generator = True
    optimizer = Optimizers.OptimizerFactory().adamax_optimizer(init_lr)
    model = Models.ModelFactory(get_num_classes(test_set)).make_basic_model()
    dataset = load_dataset_generator(test_set, test_size, batch_size) if use_generator\
              else load_dataset(test_set, test_size)
    history = calculate_for(batch_size, epochs, optimizer, model, dataset, use_generator)
    display_results(model, batch_size, history, dataset, use_generator, test_set)


if __name__ == "__main__":
    main()
