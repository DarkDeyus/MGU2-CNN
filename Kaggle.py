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
from PIL import Image
import os
import pandas as pd

def normalise(x):
    return (x / 255) - 0.5

def load_dir(path):
    files = [str(x) + ".png" for x in range(1,300001)]
    dataset = normalise(np.array([np.array(Image.open(os.path.join(path, file))) for file in files]))
    return dataset

def main():
    model_path = "data/vgg3_adamax0.001_400_TestSet.CIFAR10_gen_2/model"
    dataset = np.load("c10test.npy")
    model = keras.models.load_model(model_path)
    predicted = model.predict_classes(dataset)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    textLabels = [labels[x] for x in predicted]
    id = range(1, 300001)
    df = pd.DataFrame({"id": id, "label": textLabels})
    df.to_csv("resultsvgg3.csv", index=False)

if __name__ == "__main__":
    main()