import Optimizers
import Main
import Models
import pickle
import os
import numpy as np
import tensorflow as tf


def calculate_accuracy(model, batch_size, dataset):
    _, _, _, _, x_test, y_test = dataset
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return acc

def run(model, model_name, optimizer, optimizer_name, retry_no, epochs, batch_size, dataset, dataset_name):
    print(f"Processing {model_name} with optimizer {optimizer_name}" +
            f"for {dataset_name} - calculation {retry_no}")
    np.random.seed(retry_no)
    tf.random.set_seed(retry_no)
    dirname = f"{model_name}_{optimizer_name}_{epochs}_{dataset_name}_{retry_no}"
    fullpath = os.path.join("data", dirname)
    os.mkdir(fullpath)
    history = Main.calculate_for(batch_size, epochs, optimizer, model, dataset)
    with open(os.path.join(fullpath, "history"), "wb") as f:
        pickle.dump(history.history, f)
    acc = calculate_accuracy(model, batch_size, dataset)
    with open(os.path.join(fullpath, "accuracy"), "w") as f:
        f.write(f"{acc}")
    model.save(os.path.join(fullpath, "model"))

def main():
    epochs = 30
    batch_size = 2000
    test_size = 0.1
    retries = 3
    optimizer_factory = Optimizers.OptimizerFactory()
    optimizers_dict = {
        "adam": lambda: optimizer_factory.adam_optimizer(0.005),
        "adamax": lambda: optimizer_factory.adamax_optimizer(0.005),
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
    dataset = Main.load_cifar10(test_size)
    dataset_name = "cifar10"
    for (model_name, model_func) in models_dict.items():
        for (optimizer_name, optimizer_func) in optimizers_dict.items():
            for i in range(retries):
                i = i + 1
                model = model_func()
                optimizer = optimizer_func()
                run(model, model_name, optimizer, optimizer_name, i,
                    epochs, batch_size, dataset, dataset_name)


def main2():
    epochs = 500
    batch_size = 2000
    test_size = 0.1
    retries = 2
    optimizer_factory = Optimizers.OptimizerFactory()
    optimizers_dict = {
        "adam": lambda: optimizer_factory.adam_optimizer(0.005),
        "adamax": lambda: optimizer_factory.adamax_optimizer(0.005),
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
        ("basic", "adamax"),
        ("vgg2", "adamax"),
        ("vgg3", "adamax"),
        ("vgg3", "adadelta"),
        ("vgg3", "adam"),
        ("vgg4", "adamax")
    ]
    dataset = Main.load_cifar10(test_size)
    dataset_name = "cifar10"
    for (model_name, optimizer_name) in considered_models:
        model = models_dict[model_name]()
        optimizer = optimizers_dict[optimizer_name]()
        for i in range(retries):
            i = i + 1
            run(model, model_name, optimizer, optimizer_name, i,
                epochs, batch_size, dataset, dataset_name)


if __name__ == "__main__":
    main2()
