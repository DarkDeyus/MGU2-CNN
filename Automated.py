import Optimizers
import Main
import Models
import pickle
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def calculate_accuracy(model, batch_size, dataset, use_generator):
    if use_generator:
        _, _, x_test, y_test, _, _ = dataset
    else:
        _, _, _, _, x_test, y_test = dataset
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return acc

def save_confusion_matrix(y_pred, y_real, test_set, dirname):
    target_names = Main.get_labels(test_set)
    plot_name = "confusion_matrix"
    extension = "png"
    filename = f"{plot_name}.{extension}"
    cm = confusion_matrix(y_pred, y_real)
    cmap = plt.get_cmap('Blues')
    if test_set is Main.TestSet.CIFAR10:
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
    plt.savefig(os.path.join(dirname, filename))

def save_accuracy_plots(history, dirname):
    plot_name = "accuracy_plots"
    extension = "png"
    filename = f"{plot_name}.{extension}"
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
    plt.savefig(os.path.join(dirname, filename))

def save_plots(y_pred, y_real, test_set, history, dirname):
    save_confusion_matrix(y_pred, y_real, test_set, dirname)
    save_accuracy_plots(history, dirname)


def run(model, model_name, optimizer, optimizer_name, retry_no, epochs, batch_size, dataset, test_set, use_generator):
    dataset_name = str(test_set)
    print(f"Processing {model_name} with optimizer {optimizer_name} " +
            ("using generator" if use_generator else "without generator") +
            f" for {dataset_name} - calculation {retry_no}")
    if use_generator:
        x_train, y_train, x_test, y_test, train_generator, valid_generator = dataset
    else:
        x_train, y_train, x_validation, y_validation, x_test, y_test = dataset
    np.random.seed(retry_no)
    tf.random.set_seed(retry_no)
    generator_text = "gen" if use_generator else "nogen"
    dirname = f"{model_name}_{optimizer_name}_{epochs}_{dataset_name}_{generator_text}_{retry_no}"
    fullpath = os.path.join("data", dirname)
    os.mkdir(fullpath)
    history = Main.calculate_for(batch_size, epochs, optimizer, model, dataset, use_generator)
    with open(os.path.join(fullpath, "history"), "wb") as f:
        pickle.dump(history.history, f)
    acc = calculate_accuracy(model, batch_size, dataset, use_generator)
    with open(os.path.join(fullpath, "accuracy"), "w") as f:
        f.write(f"{acc}")
    model.save(os.path.join(fullpath, "model"))
    prediction = model.predict_classes(x_test, batch_size=batch_size)
    y_real = np.argmax(y_test, axis=1)
    save_plots(prediction, y_real, test_set, history, fullpath)

def main():
    epochs = 30
    batch_size = 2000
    test_size = 0.1
    retries = 3
    optimizer_factory = Optimizers.OptimizerFactory()
    optimizers_dict = {
        "adam": lambda: optimizer_factory.adam_optimizer(0.005),
        "adamax0.005": lambda: optimizer_factory.adamax_optimizer(0.005),
        "adamax0.001": lambda: optimizer_factory.adamax_optimizer(0.001),
        "nadam": lambda: optimizer_factory.nadam_optimizer(0.005),
        "adagrad": lambda: optimizer_factory.adagrad_optimizer(0.005),
        "adadelta": lambda: optimizer_factory.adadelta_optimizer(0.005),
        "sgd": lambda: optimizer_factory.sgd_optimizer(0.005, 0.05),
        "rmsprop": lambda: optimizer_factory.rmsprop_optimizer(0.005)
    }
    model_factory = Models.ModelFactory(10)
    models_dict = {
        "basic": lambda: model_factory.make_basic_model(),
        "vgg2": lambda: model_factory.make_vgg_model_2_block(),
        "vgg3": lambda: model_factory.make_vgg_model_3_block(),
        "vgg4": lambda: model_factory.make_vgg_model_4_block()
    }
    test_set = Main.TestSet.CIFAR10
    dataset = Main.load_dataset(test_set, test_size)
    for (model_name, model_func) in models_dict.items():
        for (optimizer_name, optimizer_func) in optimizers_dict.items():
            for i in range(retries):
                i = i + 1
                model = model_func()
                optimizer = optimizer_func()
                run(model, model_name, optimizer, optimizer_name, i,
                    epochs, batch_size, dataset, test_set, False)


def main2():
    epochs = 400
    batch_size = 2000
    test_size = 0.2
    retries = 2
    optimizer_factory = Optimizers.OptimizerFactory()
    optimizers_dict = {
        "adam": lambda: optimizer_factory.adam_optimizer(0.005),
        "adamax0.005": lambda: optimizer_factory.adamax_optimizer(0.005),
        "adamax0.001": lambda: optimizer_factory.adamax_optimizer(0.001),
        "nadam": lambda: optimizer_factory.nadam_optimizer(0.005),
        "adagrad": lambda: optimizer_factory.adagrad_optimizer(0.005),
        "adadelta": lambda: optimizer_factory.adadelta_optimizer(0.005),
        "sgd": lambda: optimizer_factory.sgd_optimizer(0.005, 0.05),
        "rmsprop": lambda: optimizer_factory.rmsprop_optimizer(0.005)
    }
    model_factory = Models.ModelFactory(10)
    models_dict = {
        "basic": lambda: model_factory.make_basic_model(),
        "vgg2": lambda: model_factory.make_vgg_model_2_block(),
        "vgg3": lambda: model_factory.make_vgg_model_3_block(),
        "vgg4": lambda: model_factory.make_vgg_model_4_block()
    }
    considered_models = [
        ("basic", "adamax0.001"),
        ("vgg2", "adamax0.001"),
        ("vgg3", "adamax0.001"),
        ("vgg4", "adamax0.001")
    ]
    datsets = [
        (Main.TestSet.CIFAR10, False),
        (Main.TestSet.CIFAR10, True),
        (Main.TestSet.CIFAR100, False),
        (Main.TestSet.CIFAR100, True)
    ]
    for datase_info in datsets:
        test_set, use_generator = datase_info
        dataset = Main.load_dataset_generator(test_set, batch_size, test_size)\
                  if use_generator else\
                  Main.load_dataset(test_set, test_size)
        for (model_name, optimizer_name) in considered_models:
            model = models_dict[model_name]()
            optimizer = optimizers_dict[optimizer_name]()
            for i in range(retries):
                i = i + 1
                run(model, model_name, optimizer, optimizer_name, i,
                    epochs, batch_size, dataset, test_set, use_generator)


if __name__ == "__main__":
    main2()
