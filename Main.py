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
import Optimizers
import Models


def load_cifar10(test_size):
    num_classes = 10
    (x_train_loaded, y_train_loaded), (x_test_loaded, y_test_loaded) = cifar10.load_data()
    # normalisation
    x_train_norm = (x_train_loaded / 255) - 0.5
    x_test = (x_test_loaded / 255) - 0.5
    # make validation set
    x_train, x_validation, y_train_to_categorical, y_validation_to_categorical = train_test_split(x_train_norm, y_train_loaded, test_size=test_size)
    # One hot encode
    y_train = keras.utils.to_categorical(y_train_to_categorical, num_classes)
    y_test = keras.utils.to_categorical(y_test_loaded, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_to_categorical, num_classes)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def set_memory_usage():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def calculate_for(batch_size, epochs, optimizer, model, dataset):
    #set_memory_usage()
    x_train, y_train, x_validation, y_validation, _, _ = dataset
    model.compile(loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=optimizer,
    metrics=['accuracy'])  # report accuracy during training

    history = model.fit(
        x_train, y_train,  # prepared data
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_validation, y_validation),  # validation_set
        shuffle=True,
        verbose=2,
        initial_epoch=0)

    return history


def display_results(model, batch_size, history, dataset):
    _, _, _, _, x_test, y_test = dataset
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Accuracy = %.2f " % (acc * 100.0) + "%")
    prediction = model.predict_classes(x_test, batch_size=batch_size)
    show_confusion_matrix(prediction, np.argmax(y_test, axis=1), Models.cifar10_classes)
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


def main():
    init_lr = 1e-3
    batch_size = 32
    epochs = 10
    test_size = 0.1
    optimizer = Optimizers.OptimizerFactory().adamax_optimizer(init_lr)
    model = Models.ModelFactory(10).make_basic_model()
    dataset = load_cifar10(test_size)
    history = calculate_for(batch_size=batch_size,
                  epochs=epochs,
                  optimizer=optimizer,
                  model=model,
                  dataset=dataset)
    display_results(model, batch_size, history, dataset)

if __name__ == "__main__":
    main()
